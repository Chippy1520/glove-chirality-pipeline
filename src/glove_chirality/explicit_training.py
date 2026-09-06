"""Maintained, import-safe explicit-split diagnostic training (never re-splits data).

The staged preset uses 23 total epochs, 3 head-only, head LR 3e-4 and
backbone LR 2e-5. Unlike the exported post-DINO script it preserves Adam moments
and keeps head LR fixed unless --fine-tune-head-learning-rate 1e-4 is supplied.
Neither this recipe nor a best checkpoint is evidence of production readiness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from glove_chirality.dataset import CLASSES, ManifestDataset
from glove_chirality.fine_tuning import FineTuning, fine_tuning_config
from glove_chirality.models import CLASSIFIER_CHOICES, build_model, model_backend
from glove_chirality.training import classification_metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, required=True)
    result.add_argument('--train-dir', type=Path, help='Default: ROOT/train')
    result.add_argument('--val-dir', type=Path, help='Default: ROOT/val')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--metrics-dir', type=Path)
    result.add_argument('--tensorboard-logdir', '--tb', type=Path)
    result.add_argument('--model', choices=CLASSIFIER_CHOICES, default='mobilenet_v3_small')
    result.add_argument('--epochs', type=int, default=None, help='Total epochs: 30 (preset: 23)')
    result.add_argument('--head-only-epochs', type=int, default=None)
    result.add_argument('--backbone-learning-rate', type=float)
    result.add_argument('--fine-tune-head-learning-rate', type=float,
                        help='Optional head LR at unfreeze; 1e-4 matches exported LR change')
    result.add_argument('--staged-finetune', action='store_true', help='23 total/3 warmup; '
                        'head 3e-4, backbone 2e-5; preserves head optimizer moments')
    result.add_argument('--batch-size', type=int, default=32)
    result.add_argument('--image-size', type=int, default=224)
    result.add_argument('--learning-rate', type=float, default=3e-4)
    result.add_argument('--weight-decay', type=float, default=1e-4)
    result.add_argument('--patience', type=int, default=8)
    result.add_argument('--workers', type=int, default=0)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--device', default='auto')
    result.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--gradient-clip-norm', type=float, default=None)
    result.add_argument('--expected-selection-sha256')
    return result


def recipe(args):
    args.epochs = args.epochs if args.epochs is not None else 23 if args.staged_finetune else 30
    if args.head_only_epochs is None:
        args.head_only_epochs = 3 if args.staged_finetune else 0
    if args.backbone_learning_rate is None and args.staged_finetune:
        args.backbone_learning_rate = 2e-5
    config = fine_tuning_config(args.epochs, args.learning_rate, args.head_only_epochs,
                               args.backbone_learning_rate)
    for name in ('batch_size', 'image_size', 'patience'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('workers and finite weight_decay must be nonnegative')
    for name in ('gradient_clip_norm', 'fine_tune_head_learning_rate'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if args.fine_tune_head_learning_rate is not None and not config['enabled']:
        raise ValueError('fine_tune_head_learning_rate requires staged fine-tuning')
    if (args.fine_tune_head_learning_rate is not None
            and args.fine_tune_head_learning_rate <= config['backbone_learning_rate']):
        raise ValueError('fine_tune_head_learning_rate must exceed backbone_learning_rate')
    return config


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inspect_split(root, train_dir=None, val_dir=None, expected=None):
    """Check bytes/membership, plus declared provenance, not inferred session isolation."""
    root = Path(root).resolve()
    directories = {'train': Path(train_dir or root / 'train').resolve(),
                   'val': Path(val_dir or root / 'val').resolve()}
    rows, pairs, provenance = {}, [], {}
    manifests = {s: root / f'manifest_{s}.csv' for s in directories}
    if any(p.exists() for p in manifests.values()) and not all(
            p.is_file() for p in manifests.values()):
        raise ValueError('Both manifest_train.csv and manifest_val.csv are required when supplied')
    checked = ['resolved_path_disjoint', 'sha256_disjoint', 'both_classes']
    for split, directory in directories.items():
        current = []
        for label in CLASSES:
            base = directory / label
            paths = sorted(p for p in base.rglob('*') if p.is_file()
                           and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            if not paths:
                raise ValueError(f'No images for {split}/{label}')
            for path in paths:
                resolved = path.resolve()
                if not resolved.is_relative_to(base.resolve()):
                    raise ValueError('Image escapes its split/class directory')
                current.append({'absolute_path': str(resolved), 'label': label,
                                'sha256': digest(resolved),
                                'image_path': path.relative_to(root).as_posix()
                                if path.is_relative_to(root) else str(path)})
        by_path = {r['absolute_path']: r for r in current}
        if len(by_path) != len(current):
            raise ValueError('Duplicate resolved image paths')
        if manifests[split].exists():
            with manifests[split].open(encoding='utf-8-sig', newline='') as stream:
                recorded = list(csv.DictReader(stream))
            seen = set()
            for row in recorded:
                key = str((root / row['image_path']).resolve())
                actual = by_path.get(key)
                if (key in seen or actual is None or actual['label'] != row['label']
                        or actual['sha256'] != row['sha256']):
                    raise ValueError(f'{split} manifest membership/label/sha256 mismatch')
                seen.add(key)
                actual['image_path'] = row['image_path']  # exact exported hash spelling
            if seen != set(by_path):
                raise ValueError(f'{split} manifest membership mismatch')
            provenance[split] = recorded
        rows[split] = current
        pairs.extend((r['image_path'], r['sha256']) for r in current)
    for field in ('absolute_path', 'sha256'):
        if {r[field] for r in rows['train']} & {r[field] for r in rows['val']}:
            raise ValueError(f'Cross-split {field} overlap')
    if provenance:
        checked.append('manifest_exact_membership_label_sha256')
        for field in ('source_group', 'source_group_id', 'source_video', 'session', 'session_id'):
            values = [{r.get(field, '').strip() for r in provenance[s]} for s in ('train', 'val')]
            if any(v - {''} for v in values):
                if (values[0] - {''}) & (values[1] - {''}):
                    raise ValueError(f'Cross-split provenance overlap: {field}')
                checked.append(f'{field}_nonempty_disjoint' +
                               ('_partial' if any('' in v for v in values) else ''))
    selection = hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest()
    if expected and selection != expected:
        raise ValueError('Expected selection SHA256 mismatch')
    return rows, {'selection_sha256': selection, 'checks': checked,
                  'manifest_sha256': {s: digest(p) for s, p in manifests.items() if p.exists()},
                  'provenance_scope': 'Declared fields only; no inferred session guarantee',
                  'counts': {s: dict(Counter(r['label'] for r in rs)) for s, rs in rows.items()}}


def distribution(predictions):
    counts = [predictions.count(i) for i in range(len(CLASSES))]
    fractions = [n / max(len(predictions), 1) for n in counts]
    return {'counts': dict(zip(CLASSES, counts, strict=True)),
            'fractions': dict(zip(CLASSES, fractions, strict=True)),
            'collapsed': bool(predictions and max(fractions) >= .99)}


def destinations(args):
    output = args.output.resolve()
    paths = [output, output.with_suffix('.history.json'), output.with_suffix('.metrics.json'),
             output.with_suffix(output.suffix + '.metrics.json')]
    paths += [p.resolve() for p in (args.metrics_dir, args.tensorboard_logdir) if p is not None]
    if len(paths) != len(set(paths)):
        raise ValueError('Output destinations collide')
    for path in paths:
        if path.exists() or path.is_symlink() or path.with_name(path.name + '.tmp').exists():
            raise FileExistsError(f'Refusing to overwrite run output: {path}')
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            raise ValueError(f'Output parent is not a directory: {path}')
    for file_path in paths[:4]:
        for directory in paths[4:]:
            if directory.is_relative_to(file_path):
                raise ValueError('Output file cannot contain an output directory')
    # A directory containing the checkpoint is useful; other directory nesting is ambiguous.
    if args.metrics_dir and output in {
            (args.metrics_dir / 'history.json').resolve(),
            (args.metrics_dir / 'best.metrics.json').resolve()}:
        raise ValueError('Checkpoint collides with metrics JSON')
    if args.metrics_dir and args.tensorboard_logdir:
        a, b = args.metrics_dir.resolve(), args.tensorboard_logdir.resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Metrics and TensorBoard directories must be separate')
    return paths[:4]


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run(args):
    started = time.time()
    config = recipe(args)
    output, history_path, metrics_path, sidecar = destinations(args)
    rows, integrity = inspect_split(args.root, args.train_dir, args.val_dir,
                                    args.expected_selection_sha256)
    summary = {'status': 'running', 'production_ready': False, 'checkpoint_role': 'diagnostic_best',
               'recipe': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'fine_tuning': config, 'model': args.model, 'model_backend': model_backend(args.model),
               'classes': CLASSES, 'image_size': args.image_size, 'augmentation': 'standard',
               'selection_metric': 'macro_recall', 'selection_tiebreaker': 'macro_f1',
               'split_id': 'explicit-' + integrity['selection_sha256'],
               'train_counts': integrity['counts']['train'],
               'validation_counts': integrity['counts']['val'],
               'train_samples': len(rows['train']), 'validation_samples': len(rows['val']),
               'training_strategy': 'staged' if config['enabled'] else 'full_model',
               'checkpoint_path': str(output),
               'dataset_selection_sha256': integrity['selection_sha256'],
               'integrity': integrity, 'history': [], 'best_validation': None}

    def persist():
        summary['training_seconds'] = time.time() - started
        summary['timing_method'] = 'run() elapsed wall clock including integrity preflight'
        for path in (history_path, metrics_path, sidecar):
            _save_json(path, summary)
        if args.metrics_dir:
            _save_json(args.metrics_dir / 'history.json', summary)
            _save_json(args.metrics_dir / 'best.metrics.json', summary)

    persist()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        summary['runtime_versions'] = {'python': platform.python_version()}
        for package in ('torch', 'torchvision', 'timm'):
            try:
                summary['runtime_versions'][package] = version(package)
            except PackageNotFoundError:
                summary['runtime_versions'][package] = None
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                              if args.device == 'auto' else args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no model weights downloaded')
        use_amp = args.amp and device.type == 'cuda'
        summary.update(device=str(device), mixed_precision=use_amp,
                       precision='cuda_float16_amp' if use_amp else 'float32')
        loaders = {s: DataLoader(ManifestDataset(rs, args.image_size, s == 'train'),
                                batch_size=args.batch_size, shuffle=s == 'train',
                                generator=torch.Generator().manual_seed(args.seed),
                                num_workers=args.workers, pin_memory=device.type == 'cuda',
                                persistent_workers=args.workers > 0) for s, rs in rows.items()}
        with ExitStack() as stack:
            writer = None
            if args.tensorboard_logdir:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(args.tensorboard_logdir))
                stack.callback(writer.close)
                writer.add_text('run/config', json.dumps(summary, default=str))
            model = build_model(args.model, num_classes=len(CLASSES), pretrained=True).to(device)
            tuning = FineTuning(model, args.model, config)
            optimizer = tuning.optimizer
            optimizer.defaults['weight_decay'] = args.weight_decay
            for group in optimizer.param_groups:
                group['weight_decay'] = args.weight_decay
            scheduler = None if config['enabled'] else torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=.5, patience=3)
            scaler = (torch.amp.GradScaler('cuda', enabled=use_amp)
                      if hasattr(torch.amp, 'GradScaler')
                      else torch.cuda.amp.GradScaler(enabled=use_amp))
            counts = integrity['counts']['train']
            weights = torch.tensor([len(rows['train']) / (len(CLASSES) * counts[c])
                                    for c in CLASSES], device=device)
            summary['class_weights'] = weights.cpu().tolist()
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            best, stale, streak, previous_stage = (-1., -1.), 0, 0, None
            for epoch in range(args.epochs):
                state = tuning.begin_epoch(epoch)
                if state['stage'] != previous_stage:
                    stale = 0
                    if state['stage'] == 'fine_tune' and args.fine_tune_head_learning_rate:
                        optimizer.param_groups[0]['lr'] = args.fine_tune_head_learning_rate
                        state['head_learning_rate'] = args.fine_tune_head_learning_rate
                previous_stage = state['stage']
                state['learning_rate'] = optimizer.param_groups[0]['lr']
                state.update(status='running', trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad),
                    optimizer_steps=0, amp_overflows=0, scaler_downscales=0, gradient_norms=[])
                summary['history'].append(state)
                for split, loader in loaders.items():
                    if split == 'val':
                        model.eval()
                    targets_all, predictions, losses = [], [], 0.
                    with torch.set_grad_enabled(split == 'train'):
                        for images, targets in loader:
                            images, targets = images.to(device), targets.to(device)
                            if split == 'train':
                                optimizer.zero_grad(set_to_none=True)
                            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=use_amp):
                                logits = model(images)
                                if not torch.isfinite(logits).all():
                                    raise RuntimeError(f'Nonfinite {split} logits')
                                loss = loss_fn(logits, targets)
                            if not torch.isfinite(loss):
                                raise RuntimeError(f'Nonfinite {split} loss')
                            if split == 'train':
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)  # diagnose actual unscaled gradients
                                grads = [p.grad for p in model.parameters() if p.grad is not None]
                                if not grads:
                                    raise RuntimeError('No gradients')
                                # Reduce on-device, avoiding a CUDA synchronization per parameter.
                                finite = bool(torch.stack([torch.isfinite(g).all()
                                                           for g in grads]).all())
                                norm = float(torch.stack([g.detach().double().square().sum()
                                                          for g in grads]).sum().sqrt()) if finite else None
                                state['gradient_norms'].append(norm)
                                if not finite and not use_amp:
                                    raise RuntimeError('Nonfinite FP32 gradients; optimizer not stepped')
                                if finite and args.gradient_clip_norm:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                   args.gradient_clip_norm,
                                                                   error_if_nonfinite=True)
                                old_scale = scaler.get_scale()
                                scaler.step(optimizer)  # GradScaler skips nonfinite AMP updates
                                scaler.update()
                                downscaled = scaler.get_scale() < old_scale
                                state['scaler_downscales'] += int(downscaled)
                                state['amp_overflows'] += int(not finite)
                                state['optimizer_steps'] += int(finite and not downscaled)
                            losses += float(loss.detach()) * len(targets)
                            targets_all.extend(targets.detach().cpu().tolist())
                            predictions.extend(logits.detach().argmax(1).cpu().tolist())
                    metrics = classification_metrics(targets_all, predictions)
                    metrics.update(loss=losses / len(targets_all),
                                   prediction_distribution=distribution(predictions))
                    state['train' if split == 'train' else 'validation'] = metrics
                    if split == 'train':
                        state['train_loss'] = metrics['loss']
                    if split == 'train' and not state['optimizer_steps']:
                        raise RuntimeError('Entire epoch had no optimizer steps')
                metrics = state['validation']
                score = (metrics['macro_recall'], metrics['macro_f1'])
                if score > best:
                    best, stale = score, 0
                    summary.update(best_validation=metrics, best_epoch=epoch + 1,
                                   best_macro_recall=best[0], best_macro_f1=best[1])
                    torch.save({'state_dict': model.state_dict(), 'model_name': args.model,
                                'model_backend': model_backend(args.model), 'classes': CLASSES,
                                'image_size': args.image_size, 'fine_tuning': config,
                                'preprocessing': 'imagenet_rgb_normalized_no_reflection',
                                'selection_metric': 'macro_recall', 'augmentation': 'standard',
                                'training_loss': 'weighted_cross_entropy', 'seed': args.seed,
                                'dataset_selection_sha256': integrity['selection_sha256'],
                                'validation_metrics': metrics, 'recipe': summary['recipe'],
                                'precision': summary['precision'],
                                'runtime_versions': summary['runtime_versions'],
                                'class_weights': summary['class_weights'],
                                'head_learning_rate': state['head_learning_rate'],
                                'backbone_learning_rate': state['backbone_learning_rate'],
                                'trainable_parameters': state['trainable_parameters'],
                                'integrity': integrity,
                                'production_ready': False, 'checkpoint_role': 'diagnostic_best',
                                'epoch': epoch + 1, 'best_epoch': epoch + 1,
                                'stage': state['stage']}, output)
                else:
                    stale += 1
                warmup = state['stage'] == 'head_only'
                streak = streak + 1 if not warmup and metrics[
                    'prediction_distribution']['collapsed'] else 0
                state.update(status='finished', collapse_streak=streak, scaler_scale=scaler.get_scale())
                if scheduler:
                    scheduler.step(score[0])
                if writer:
                    writer.add_text('training/stage', state['stage'], epoch + 1)
                    for key in ('head_learning_rate', 'backbone_learning_rate',
                                'optimizer_steps', 'amp_overflows', 'scaler_downscales',
                                'scaler_scale', 'trainable_parameters', 'collapse_streak'):
                        writer.add_scalar(f'training/{key}', state[key], epoch + 1)
                    norms = [v for v in state['gradient_norms'] if v is not None]
                    if norms:
                        writer.add_scalar('training/gradient_norm_max', max(norms), epoch + 1)
                    for split in ('train', 'validation'):
                        for key in ('loss', 'accuracy', 'macro_recall', 'macro_f1'):
                            writer.add_scalar(f'{split}/{key}', state[split][key], epoch + 1)
                        for index, label in enumerate(CLASSES):
                            writer.add_scalar(f'{split}/recall_{label}',
                                              state[split]['recall_per_class'][index], epoch + 1)
                            writer.add_scalar(f'{split}/predicted_fraction_{label}', state[split][
                                'prediction_distribution']['fractions'][label], epoch + 1)
                    writer.flush()
                persist()
                print(f"epoch={epoch + 1} stage={state['stage']} macro_recall={score[0]:.4f} "
                      f"macro_f1={score[1]:.4f} collapse_streak={streak}")
                if streak >= 3:
                    summary['status'] = 'collapsed'
                    break
                if not warmup and stale >= args.patience:
                    summary['stop_reason'] = 'early_stopping'
                    break
            if summary['status'] != 'collapsed':
                best_metrics = summary['best_validation']
                summary['status'] = ('not_converged' if best[0] <= .5 or best_metrics[
                    'prediction_distribution']['collapsed'] else 'completed_diagnostic')
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if summary['history'] and summary['history'][-1]['status'] == 'running':
            summary['history'][-1]['status'] = 'failed'
        raise
    finally:
        try:
            _, end = inspect_split(args.root, args.train_dir, args.val_dir,
                                   integrity['selection_sha256'])
            if end != integrity:
                raise ValueError('Integrity metadata changed during training')
            summary['integrity_rechecked'] = True
        except Exception as exc:  # noqa: BLE001 - preserve original failure and record recheck failure
            summary.update(status='failed', integrity_rechecked=False,
                           integrity_error=f'{type(exc).__name__}: {exc}')
        persist()
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI converts runtime failures to nonzero status
        print(f'Training failed: {exc}')
        return 1
    print(f"status={summary['status']}; checkpoint is diagnostic, not production-ready")
    return 0 if summary['status'] == 'completed_diagnostic' else 1


if __name__ == '__main__':
    raise SystemExit(main())
