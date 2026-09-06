"""Synthetic safety coverage only: no pretrained downloads or accuracy claims."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from glove_chirality import explicit_training as explicit


def data(tmp_path):
    root = tmp_path / 'data'
    for split in ('train', 'val'):
        for label in ('left', 'right'):
            folder = root / split / label / ('nested' if label == 'right' else '')
            folder.mkdir(parents=True)
            (folder / 'crop.PNG').write_bytes(f'{split}/{label}'.encode())
    return root


def options(root, tmp_path, *extra):
    return explicit.parser().parse_args(['--root', str(root), '--output',
                                        str(tmp_path / 'run' / 'best.pt'), *extra])


def manifests(root, provenance=False):
    rows, _ = explicit.inspect_split(root)
    for split, values in rows.items():
        with (root / f'manifest_{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_path', 'label', 'sha256',
                                                       'source_video'])
            writer.writeheader()
            for row in values:
                writer.writerow({**{k: row[k] for k in ('image_path', 'label', 'sha256')},
                                 'source_video': 'shared' if provenance else split})
    return rows


def test_defaults_and_preset(tmp_path):
    args = options(tmp_path, tmp_path)
    config = explicit.recipe(args)
    assert args.epochs == 30 and args.learning_rate == 3e-4
    assert args.weight_decay == 1e-4 and args.patience == 8 and args.amp
    assert not config['enabled'] and args.gradient_clip_norm is None
    args = options(tmp_path, tmp_path, '--staged-finetune', '--no-amp',
                   '--model', 'dinov3_vit_small', '--fine-tune-head-learning-rate', '0.0001')
    config = explicit.recipe(args)
    assert args.epochs == 23 and args.head_only_epochs == 3
    assert config['backbone_learning_rate'] == 2e-5 and not args.amp


def test_wrapper_import_safe():
    path = Path(__file__).parents[1] / 'scripts' / 'train_layer2_explicit_split.py'
    spec = importlib.util.spec_from_file_location('explicit_wrapper', path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_integrity_and_export_hash(tmp_path):
    root = data(tmp_path)
    rows = manifests(root)
    pairs = [(r['image_path'], r['sha256']) for rs in rows.values() for r in rs]
    import hashlib
    expected = hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest()
    _, report = explicit.inspect_split(root, expected=expected)
    assert report['selection_sha256'] == expected
    assert 'source_video_nonempty_disjoint' in report['checks']
    with pytest.raises(ValueError, match='Expected selection'):
        explicit.inspect_split(root, expected='bad')
    Path(rows['train'][0]['absolute_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest'):
        explicit.inspect_split(root)


def test_overlap_missing_class_provenance(tmp_path):
    root = data(tmp_path)
    manifests(root, provenance=True)
    with pytest.raises(ValueError, match='provenance overlap'):
        explicit.inspect_split(root)
    for path in root.glob('manifest*'):
        path.unlink()
    rows, _ = explicit.inspect_split(root)
    Path(rows['val'][0]['absolute_path']).write_bytes(Path(rows['train'][0]['absolute_path']).read_bytes())
    with pytest.raises(ValueError, match='sha256 overlap'):
        explicit.inspect_split(root)
    Path(rows['val'][0]['absolute_path']).unlink()
    with pytest.raises(ValueError, match='No images'):
        explicit.inspect_split(root)


@pytest.mark.parametrize('destination', ['output', 'history', 'metrics', 'sidecar', 'tb', 'metrics_dir'])
def test_refuse_overwrite_before_model(tmp_path, monkeypatch, destination):
    args = options(data(tmp_path), tmp_path, '--tb', str(tmp_path / 'tb'),
                   '--metrics-dir', str(tmp_path / 'metrics'))
    targets = {'output': args.output, 'history': args.output.with_suffix('.history.json'),
               'metrics': args.output.with_suffix('.metrics.json'),
               'sidecar': args.output.with_suffix('.pt.metrics.json'),
               'tb': args.tensorboard_logdir, 'metrics_dir': args.metrics_dir}
    target = targets[destination]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('keep')
    monkeypatch.setattr(explicit, 'build_model', lambda *a, **k: pytest.fail('model loaded'))
    with pytest.raises(FileExistsError):
        explicit.run(args)
    assert target.read_text() == 'keep'


def tiny_fixture(tmp_path, monkeypatch, *, collapsed=False, nonfinite=False, bad_grad=False):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from PIL import Image
    root = data(tmp_path)
    for index, path in enumerate(sorted(root.rglob('*.PNG'))):
        Image.new('RGB', (8, 8), (30 + index * 40, 40, 60)).save(path)
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    if collapsed:
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.copy_(torch.tensor([10., -10.]))
    if nonfinite:
        model.register_forward_hook(lambda m, a, out: out * float('nan'))
    if bad_grad:
        model[-1].weight.register_hook(lambda grad: grad * float('nan'))
    monkeypatch.setattr(explicit, 'build_model', lambda *a, **k: model)
    args = options(root, tmp_path, '--model', 'tiny_cnn', '--device', 'cpu', '--no-amp',
                   '--epochs', '2', '--head-only-epochs', '1', '--batch-size', '2',
                   '--image-size', '8', '--fine-tune-head-learning-rate', '0.0001')
    return args, model


def test_real_synthetic_two_stage(tmp_path, monkeypatch):
    args, _model = tiny_fixture(tmp_path, monkeypatch)
    observations = []
    original = explicit.FineTuning.begin_epoch

    def begin(self, epoch):
        result = original(self, epoch)
        observations.append((id(self.optimizer), len(self.optimizer.state),
                             [g['weight_decay'] for g in self.optimizer.param_groups]))
        return result

    monkeypatch.setattr(explicit.FineTuning, 'begin_epoch', begin)
    result = explicit.run(args)
    assert observations[0][0] == observations[1][0]
    assert observations[1][1] > 0  # retained head Adam moments at transition
    assert all(decay == 1e-4 for _, _, groups in observations for decay in groups)
    first, second = result['history']
    assert first['stage'] == 'head_only' and second['stage'] == 'fine_tune'
    assert first['trainable_parameters'] < second['trainable_parameters']
    assert second['head_learning_rate'] == 1e-4
    assert first['optimizer_steps'] == 1 and second['optimizer_steps'] == 1
    assert second['gradient_norms'] and 'prediction_distribution' in second['validation']
    assert result['integrity_rechecked'] and result['production_ready'] is False
    import torch
    checkpoint = torch.load(args.output, weights_only=True)
    assert checkpoint['model_name'] == 'tiny_cnn'
    assert checkpoint['dataset_selection_sha256'] == result['integrity']['selection_sha256']
    assert json.loads(args.output.with_suffix('.history.json').read_text()) == result


@pytest.mark.parametrize('kind', ['nonfinite', 'bad_grad'])
def test_numerical_failure_persisted(tmp_path, monkeypatch, kind):
    args, _ = tiny_fixture(tmp_path, monkeypatch, **{kind: True})
    with pytest.raises(RuntimeError, match='Nonfinite'):
        explicit.run(args)
    report = json.loads(args.output.with_suffix('.metrics.json').read_text())
    assert report['status'] == 'failed' and report['integrity_rechecked']
    assert report['history'][0]['optimizer_steps'] == 0
    assert not args.output.exists()


def test_collapse_stops_after_three_nonwarmup_epochs(tmp_path, monkeypatch):
    args, _ = tiny_fixture(tmp_path, monkeypatch, collapsed=True)
    args.epochs = 8
    report = explicit.run(args)
    assert report['status'] == 'collapsed'
    assert len(report['history']) == 4
    assert report['history'][0]['collapse_streak'] == 0
    assert report['history'][-1]['collapse_streak'] == 3


def test_minimal_import_without_ml():
    import subprocess
    import sys
    code = '''
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'timm', 'numpy', 'PIL'}:
            raise ImportError('ML deliberately unavailable')
sys.meta_path.insert(0, BlockML())
from glove_chirality.explicit_training import parser
assert 'vit_b_16' in parser().format_help()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr


def test_validation_nonfinite_and_tensorboard_closes(tmp_path, monkeypatch):
    args, model = tiny_fixture(tmp_path, monkeypatch)
    tb = pytest.importorskip('torch.utils.tensorboard')
    closed = []

    class Writer:
        def __init__(self, *args):
            pass

        def add_text(self, *args):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(tb, 'SummaryWriter', Writer)
    args.tensorboard_logdir = tmp_path / 'tb'
    # Warm-up uses model.eval(), so distinguish validation by gradient context.
    import torch
    model.register_forward_hook(lambda m, a, out: out if torch.is_grad_enabled()
                                else out * float('nan'))
    with pytest.raises(RuntimeError, match='Nonfinite val logits'):
        explicit.run(args)
    assert closed == [True]
    report = json.loads(args.output.with_suffix('.metrics.json').read_text())
    assert report['history'][0]['optimizer_steps'] == 1
    assert report['status'] == 'failed'


def test_end_integrity_recheck(tmp_path, monkeypatch):
    args, model = tiny_fixture(tmp_path, monkeypatch)
    path = next((args.root / 'train').rglob('*.PNG'))
    model.register_forward_hook(lambda m, a, out: (path.write_bytes(b'changed'), out)[1])
    # The first batch has been decoded before mutation; later epochs may also fail decoding.
    args.epochs = 1
    args.head_only_epochs = 0
    args.fine_tune_head_learning_rate = None
    report = explicit.run(args)
    assert report['status'] == 'failed' and not report['integrity_rechecked']


def test_cli_failure_status(tmp_path, monkeypatch):
    monkeypatch.setattr(explicit, 'run', lambda args: {'status': 'collapsed'})
    assert explicit.main(['--root', str(tmp_path), '--output', str(tmp_path / 'best.pt')]) == 1


def test_invalid_options(tmp_path):
    args = options(tmp_path, tmp_path, '--epochs', '2', '--head-only-epochs', '2')
    with pytest.raises(ValueError):
        explicit.recipe(args)
