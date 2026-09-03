from __future__ import annotations

import argparse
import queue
import subprocess
import threading
from pathlib import Path

import yaml

from glove_chirality import gui_commands
from glove_chirality.config import ExtractionConfig
from glove_chirality.models import CLASSIFIER_CHOICES

CUSTOM_YOLO_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}


def tight_detection_crop_preset() -> dict[str, object]:
    """Use the selected detector box without padding or square expansion."""
    return {
        "crop_padding": 0.0,
        "make_square": False,
        "crop_mode": "bbox",
    }


def custom_yolo_segmentation_preset(model_path: str | Path) -> dict[str, object]:
    """Return safe Layer-1 defaults for a selected custom glove segmenter."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Custom YOLO model not found: {path}")
    if path.suffix.lower() not in CUSTOM_YOLO_SUFFIXES:
        supported = ", ".join(sorted(CUSTOM_YOLO_SUFFIXES))
        raise ValueError(f"Unsupported YOLO model format {path.suffix!r}; use {supported}")
    return {
        "backend": "yolo",
        "yolo_model": str(path.resolve()),
        "yolo_class_id": 0,
        "yolo_use_masks": True,
        "yolo_require_masks": True,
        "yolo_crop_to_roi": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Glove Chirality Pipeline GUI")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.process: subprocess.Popen[str] | None = None
            self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
            self.root.title("Glove Chirality Pipeline")
            self.root.geometry("1180x900")
            self.root.minsize(980, 760)
            self._style(ttk)

            default_config = Path("configs/default.yaml")
            self.config_path = tk.StringVar(
                value=str(default_config.resolve()) if default_config.exists() else ""
            )
            self.notebook = ttk.Notebook(root)
            self.notebook.pack(fill="both", expand=True, padx=12, pady=(12, 6))
            self._extraction_tab(tk, ttk, filedialog, messagebox)
            self._settings_tab(tk, ttk, filedialog, messagebox)
            self._training_tab(tk, ttk, filedialog, messagebox)
            self._inference_tab(tk, ttk, filedialog, messagebox)
            self._log_panel(tk, ttk, scrolledtext)
            self.root.after(100, self._drain_messages)

        @staticmethod
        def _style(ttk_module) -> None:
            style = ttk_module.Style()
            for theme in ("vista", "clam"):
                if theme in style.theme_names():
                    style.theme_use(theme)
                    break
            style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
            style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
            style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

        @staticmethod
        def _entry(parent, ttk_module, row, label, variable, width=72, secret=False):
            ttk_module.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
            entry = ttk_module.Entry(parent, textvariable=variable, width=width, show="*" if secret else "")
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            parent.columnconfigure(1, weight=1)
            return entry

        def _path_row(
            self,
            parent,
            ttk_module,
            filedialog_module,
            row,
            label,
            variable,
            mode,
            on_selected=None,
        ):
            self._entry(parent, ttk_module, row, label, variable)

            def browse():
                if mode == "directory":
                    value = filedialog_module.askdirectory()
                elif mode == "save-csv":
                    value = filedialog_module.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
                elif mode == "save-model":
                    value = filedialog_module.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")])
                elif mode == "save-yaml":
                    value = filedialog_module.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
                elif mode == "save-image":
                    value = filedialog_module.asksaveasfilename(defaultextension=".jpg", filetypes=[("Images", "*.jpg *.png"), ("All files", "*.*")])
                elif mode == "save-jsonl":
                    value = filedialog_module.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")])
                elif mode in {"video", "video-or-directory"}:
                    value = filedialog_module.askopenfilename(filetypes=[("Videos", "*.mkv *.avi *.mp4 *.mov *.m4v"), ("All files", "*.*")])
                elif mode == "image-or-directory":
                    value = filedialog_module.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
                elif mode == "model":
                    value = filedialog_module.askopenfilename(filetypes=[("PyTorch", "*.pt *.pth"), ("All files", "*.*")])
                elif mode == "detector-model":
                    value = filedialog_module.askopenfilename(
                        filetypes=[
                            ("Ultralytics detector", "*.pt *.pth *.onnx *.engine *.torchscript"),
                            ("All files", "*.*"),
                        ]
                    )
                elif mode == "yaml":
                    value = filedialog_module.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
                elif mode == "csv":
                    value = filedialog_module.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
                else:
                    value = filedialog_module.askopenfilename()
                if value:
                    variable.set(value)
                    if on_selected is not None:
                        on_selected(value)

            ttk_module.Button(parent, text="Browse…", command=browse).grid(row=row, column=2, padx=8, pady=5)
            if mode in {"video-or-directory", "image-or-directory"}:
                def browse_folder():
                    value = filedialog_module.askdirectory()
                    if value:
                        variable.set(value)

                ttk_module.Button(parent, text="Folder…", command=browse_folder).grid(row=row, column=3, padx=(0, 8), pady=5)

        def _config_row(self, parent, ttk_module, filedialog_module, row=0):
            self._path_row(parent, ttk_module, filedialog_module, row, "Extraction config", self.config_path, "yaml")

        def _extraction_tab(self, tk_module, ttk_module, filedialog_module, messagebox_module):
            tab = ttk_module.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text="Extract")
            ttk_module.Label(tab, text="Video → representative glove crops", style="Title.TLabel").pack(anchor="w", pady=(0, 8))

            dataset = ttk_module.LabelFrame(tab, text="Labeled dataset from known streams", style="Section.TLabelframe", padding=8)
            dataset.pack(fill="x", pady=5)
            self.left_dir, self.right_dir, self.dataset_output = tk_module.StringVar(), tk_module.StringVar(), tk_module.StringVar()
            self._path_row(dataset, ttk_module, filedialog_module, 0, "Left-only videos", self.left_dir, "directory")
            self._path_row(dataset, ttk_module, filedialog_module, 1, "Right-only videos", self.right_dir, "directory")
            self._path_row(dataset, ttk_module, filedialog_module, 2, "Dataset output", self.dataset_output, "directory")
            self._config_row(dataset, ttk_module, filedialog_module, 3)
            ttk_module.Button(dataset, text="Extract labeled dataset", style="Run.TButton", command=lambda: self._guard(messagebox_module, lambda: gui_commands.extract_dataset(self.left_dir.get(), self.right_dir.get(), self.dataset_output.get(), self.config_path.get()))).grid(row=4, column=1, sticky="e", padx=8, pady=8)

            single = ttk_module.LabelFrame(tab, text="Single video or directory", style="Section.TLabelframe", padding=8)
            single.pack(fill="x", pady=5)
            self.single_input, self.single_output = tk_module.StringVar(), tk_module.StringVar()
            self.single_label = tk_module.StringVar(value="unknown")
            self._path_row(single, ttk_module, filedialog_module, 0, "Input video/directory", self.single_input, "video-or-directory")
            self._path_row(single, ttk_module, filedialog_module, 1, "Output directory", self.single_output, "directory")
            ttk_module.Label(single, text="Known label").grid(row=2, column=0, sticky="w", padx=8, pady=5)
            ttk_module.Combobox(single, textvariable=self.single_label, values=("unknown", "left", "right"), state="readonly", width=16).grid(row=2, column=1, sticky="w", padx=8, pady=5)
            ttk_module.Button(single, text="Extract events", style="Run.TButton", command=lambda: self._guard(messagebox_module, lambda: gui_commands.extract_single(self.single_input.get(), self.single_output.get(), self.single_label.get(), self.config_path.get()))).grid(row=3, column=1, sticky="e", padx=8, pady=8)

            preview = ttk_module.LabelFrame(tab, text="Calibration preview", style="Section.TLabelframe", padding=8)
            preview.pack(fill="x", pady=5)
            self.preview_video, self.preview_output = tk_module.StringVar(), tk_module.StringVar()
            self.preview_seconds = tk_module.DoubleVar(value=0.0)
            self.preview_warmup = tk_module.DoubleVar(value=2.0)
            self._path_row(preview, ttk_module, filedialog_module, 0, "Video", self.preview_video, "video")
            self._path_row(preview, ttk_module, filedialog_module, 1, "Preview image", self.preview_output, "save-image")
            self._entry(preview, ttk_module, 2, "Timestamp (seconds)", self.preview_seconds, width=16)
            self._entry(preview, ttk_module, 3, "Detector warm-up (seconds)", self.preview_warmup, width=16)
            ttk_module.Button(preview, text="Render preview", command=lambda: self._guard(messagebox_module, lambda: gui_commands.preview(self.preview_video.get(), self.preview_output.get(), self.preview_seconds.get(), self.config_path.get(), self.preview_warmup.get()))).grid(row=4, column=1, sticky="e", padx=8, pady=8)

        def _settings_tab(self, tk_module, ttk_module, filedialog_module, messagebox_module):
            tab = ttk_module.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text="Extraction settings")
            ttk_module.Label(tab, text="Layer 1 detector, event selection, and crop settings", style="Title.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
            self._path_row(tab, ttk_module, filedialog_module, 1, "Config file", self.config_path, "yaml")

            self.setting_vars = {
                "backend": tk_module.StringVar(value="belt_foreground"),
                "roi": tk_module.StringVar(value="0.05, 0.05, 0.95, 0.95"),
                "trigger_zone": tk_module.StringVar(value="0.20, 0.15, 0.80, 0.85"),
                "require_full_containment": tk_module.BooleanVar(value=True),
                "trigger_inner_margin_ratio": tk_module.DoubleVar(value=0.0),
                "color_distance_threshold": tk_module.DoubleVar(value=28.0),
                "motion_assist": tk_module.BooleanVar(value=True),
                "adaptive_background": tk_module.BooleanVar(value=True),
                "mog_empty_learning_rate": tk_module.DoubleVar(value=0.02),
                "mog_foreground_learning_rate": tk_module.DoubleVar(value=0.0),
                "morph_kernel": tk_module.IntVar(value=11),
                "min_area_ratio": tk_module.DoubleVar(value=0.015),
                "max_area_ratio": tk_module.DoubleVar(value=0.55),
                "yolo_model": tk_module.StringVar(value=""),
                "yolo_confidence": tk_module.DoubleVar(value=0.35),
                "yolo_class_id": tk_module.IntVar(value=0),
                "yolo_device": tk_module.StringVar(value="auto"),
                "yolo_half": tk_module.BooleanVar(value=False),
                "yolo_imgsz": tk_module.IntVar(value=640),
                "yolo_iou": tk_module.DoubleVar(value=0.50),
                "yolo_max_det": tk_module.IntVar(value=5),
                "yolo_min_box_area_ratio": tk_module.DoubleVar(value=0.0),
                "yolo_max_box_area_ratio": tk_module.DoubleVar(value=1.0),
                "yolo_use_masks": tk_module.BooleanVar(value=True),
                "yolo_require_masks": tk_module.BooleanVar(value=False),
                "yolo_crop_to_roi": tk_module.BooleanVar(value=False),
                "min_detected_frames": tk_module.IntVar(value=2),
                "reject_multiple_detections": tk_module.BooleanVar(value=True),
                "exit_missing_frames": tk_module.IntVar(value=5),
                "cooldown_frames": tk_module.IntVar(value=8),
                "crop_padding": tk_module.DoubleVar(value=0.12),
                "crop_mode": tk_module.StringVar(value="bbox"),
                "output_size": tk_module.IntVar(value=256),
                "make_square": tk_module.BooleanVar(value=True),
            }
            detector = ttk_module.LabelFrame(tab, text="Detector", style="Section.TLabelframe", padding=8)
            detector.grid(row=2, column=0, sticky="nsew", padx=(0, 5), pady=5)
            event = ttk_module.LabelFrame(tab, text="Passage and crop", style="Section.TLabelframe", padding=8)
            event.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
            yolo = ttk_module.LabelFrame(tab, text="Layer 1 — custom YOLO segmentation", style="Section.TLabelframe", padding=8)
            yolo.grid(row=2, column=2, sticky="nsew", padx=(5, 0), pady=5)
            tab.columnconfigure((0, 1, 2), weight=1)

            labels = [
                ("Backend", "backend"), ("ROI x1,y1,x2,y2", "roi"),
                ("Trigger x1,y1,x2,y2", "trigger_zone"),
                ("Trigger inner margin ratio", "trigger_inner_margin_ratio"),
                ("Color-distance threshold", "color_distance_threshold"),
                ("Empty-belt learning rate", "mog_empty_learning_rate"),
                ("Foreground learning rate", "mog_foreground_learning_rate"),
                ("Morphology kernel", "morph_kernel"),
                ("Minimum area ratio", "min_area_ratio"),
                ("Maximum area ratio", "max_area_ratio"),
            ]
            for row, (label, key) in enumerate(labels):
                ttk_module.Label(detector, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=5)
                if key == "backend":
                    widget = ttk_module.Combobox(detector, textvariable=self.setting_vars[key], values=("belt_foreground", "dark_contour", "yolo"), state="readonly")
                else:
                    widget = ttk_module.Entry(detector, textvariable=self.setting_vars[key], width=28)
                widget.grid(row=row, column=1, sticky="ew", padx=6, pady=5)
            ttk_module.Checkbutton(detector, text="Temporal motion assistance", variable=self.setting_vars["motion_assist"]).grid(row=len(labels), column=0, columnspan=2, sticky="w", padx=6, pady=5)
            ttk_module.Checkbutton(detector, text="Adapt background during empty gaps", variable=self.setting_vars["adaptive_background"]).grid(row=len(labels) + 1, column=0, columnspan=2, sticky="w", padx=6, pady=5)
            ttk_module.Checkbutton(detector, text="Require glove fully inside trigger", variable=self.setting_vars["require_full_containment"]).grid(row=len(labels) + 2, column=0, columnspan=2, sticky="w", padx=6, pady=5)
            detector.columnconfigure(1, weight=1)

            event_labels = [
                ("Confirmation frames", "min_detected_frames"),
                ("Missing frames to close", "exit_missing_frames"),
                ("Cooldown frames", "cooldown_frames"),
                ("Crop padding fraction", "crop_padding"),
                ("Export image size", "output_size"),
            ]
            for row, (label, key) in enumerate(event_labels):
                self._entry(event, ttk_module, row, label, self.setting_vars[key], width=20)
            ttk_module.Checkbutton(event, text="Make square crop before export", variable=self.setting_vars["make_square"]).grid(row=len(event_labels), column=0, columnspan=2, sticky="w", padx=6, pady=5)
            ttk_module.Checkbutton(event, text="Reject frames with multiple candidates", variable=self.setting_vars["reject_multiple_detections"]).grid(row=len(event_labels) + 1, column=0, columnspan=2, sticky="w", padx=6, pady=5)
            ttk_module.Label(event, text="Crop mode").grid(row=len(event_labels) + 2, column=0, sticky="w", padx=6, pady=5)
            ttk_module.Combobox(event, textvariable=self.setting_vars["crop_mode"], values=("bbox", "masked", "masked_fill"), state="readonly", width=16).grid(row=len(event_labels) + 2, column=1, sticky="w", padx=6, pady=5)
            def apply_tight_crop():
                for key, value in tight_detection_crop_preset().items():
                    self.setting_vars[key].set(value)

            ttk_module.Button(
                event,
                text="Use tight detection bbox",
                command=apply_tight_crop,
            ).grid(row=len(event_labels) + 3, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4))
            ttk_module.Label(
                event,
                text="Removes crop padding and square expansion; output is still aspect-preserving letterboxed.",
                wraplength=300,
                foreground="#3b5f7a",
            ).grid(row=len(event_labels) + 4, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))

            self.yolo_status = tk_module.StringVar(
                value="Choose the trained glove segmentation checkpoint (usually best.pt)."
            )

            def apply_custom_yolo(value=None):
                try:
                    settings = custom_yolo_segmentation_preset(
                        value or self.setting_vars["yolo_model"].get()
                    )
                    for key, setting in settings.items():
                        self.setting_vars[key].set(setting)
                    self.yolo_status.set(
                        "Custom Layer 1 enabled: class 0 = glove, masks required, "
                        "ROI-only inference. Save settings, then validate with Calibration preview."
                    )
                except (OSError, TypeError, ValueError) as exc:
                    messagebox_module.showerror("Could not apply custom detector", str(exc))

            self._path_row(
                yolo,
                ttk_module,
                filedialog_module,
                0,
                "Custom model",
                self.setting_vars["yolo_model"],
                "detector-model",
                on_selected=apply_custom_yolo,
            )
            yolo_labels = [
                ("Confidence", "yolo_confidence"),
                ("Glove class ID", "yolo_class_id"),
                ("Device", "yolo_device"),
                ("Image size", "yolo_imgsz"),
                ("IoU threshold", "yolo_iou"),
                ("Maximum detections", "yolo_max_det"),
                ("Minimum box area ratio", "yolo_min_box_area_ratio"),
                ("Maximum box area ratio", "yolo_max_box_area_ratio"),
            ]
            for row, (label, key) in enumerate(yolo_labels, 1):
                self._entry(yolo, ttk_module, row, label, self.setting_vars[key], width=18)
            for row, (label, key) in enumerate([
                ("Half precision", "yolo_half"),
                ("Use segmentation masks", "yolo_use_masks"),
                ("Require segmentation masks", "yolo_require_masks"),
                ("Run YOLO on ROI only", "yolo_crop_to_roi"),
            ], len(yolo_labels) + 1):
                ttk_module.Checkbutton(yolo, text=label, variable=self.setting_vars[key]).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)
            preset_row = len(yolo_labels) + 5
            ttk_module.Button(
                yolo,
                text="Apply as custom Layer 1 model",
                command=apply_custom_yolo,
            ).grid(row=preset_row, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 4))
            ttk_module.Label(
                yolo,
                textvariable=self.yolo_status,
                wraplength=300,
                foreground="#3b5f7a",
            ).grid(row=preset_row + 1, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 6))

            buttons = ttk_module.Frame(tab)
            buttons.grid(row=3, column=0, columnspan=3, sticky="e", pady=10)

            def save_as():
                value = filedialog_module.asksaveasfilename(
                    defaultextension=".yaml",
                    filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
                )
                if value:
                    self.config_path.set(value)
                    self._save_settings(messagebox_module)

            ttk_module.Button(buttons, text="Load YAML", command=lambda: self._load_settings(messagebox_module)).pack(side="left", padx=5)
            ttk_module.Button(buttons, text="Save settings", style="Run.TButton", command=lambda: self._save_settings(messagebox_module)).pack(side="left", padx=5)
            ttk_module.Button(buttons, text="Save as…", command=save_as).pack(side="left", padx=5)
            if self.config_path.get():
                self._load_settings(messagebox_module, quiet=True)

        def _training_tab(self, tk_module, ttk_module, filedialog_module, messagebox_module):
            tab = ttk_module.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text="Train")
            ttk_module.Label(tab, text="Train an interchangeable chirality classifier", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
            self.train_manifest, self.train_output = tk_module.StringVar(), tk_module.StringVar()
            self.train_model = tk_module.StringVar(value="resnet18")
            self.train_device = tk_module.StringVar(value="auto")
            self.train_epochs, self.train_batch, self.train_size = tk_module.IntVar(value=20), tk_module.IntVar(value=32), tk_module.IntVar(value=224)
            self.train_lr, self.train_val = tk_module.DoubleVar(value=0.001), tk_module.DoubleVar(value=0.2)
            self.train_seed, self.train_workers = tk_module.IntVar(value=42), tk_module.IntVar(value=0)
            self.train_amp = tk_module.BooleanVar(value=False)
            self.train_loss = tk_module.StringVar(value="weighted_cross_entropy")
            self.train_recall_target = tk_module.StringVar(value="right")
            self.train_recall_weight = tk_module.DoubleVar(value=1.0)
            self.train_selection_metric = tk_module.StringVar(value="macro_recall")
            self._path_row(tab, ttk_module, filedialog_module, 1, "Dataset manifest", self.train_manifest, "csv")
            self._path_row(tab, ttk_module, filedialog_module, 2, "Checkpoint output", self.train_output, "save-model")
            fields = [
                ("Model", self.train_model, CLASSIFIER_CHOICES),
                ("Device", self.train_device, ("auto", "cpu", "cuda", "cuda:0", "cuda:1")),
                ("Epochs", self.train_epochs, None), ("Batch size", self.train_batch, None),
                ("Image size", self.train_size, None), ("Learning rate", self.train_lr, None),
                ("Validation fraction", self.train_val, None), ("Seed", self.train_seed, None),
                ("DataLoader workers", self.train_workers, None),
                ("Loss", self.train_loss, ("cross_entropy", "weighted_cross_entropy", "recall_hybrid")),
                ("Recall target", self.train_recall_target, ("left", "right")),
                ("Recall penalty weight", self.train_recall_weight, None),
                ("Best-checkpoint metric", self.train_selection_metric, ("accuracy", "macro_recall", "macro_f1", "recall_left", "recall_right")),
            ]
            for index, (label, variable, values) in enumerate(fields):
                row, column = 3 + index // 2, (index % 2) * 2
                ttk_module.Label(tab, text=label).grid(row=row, column=column, sticky="w", padx=8, pady=6)
                widget = ttk_module.Combobox(tab, textvariable=variable, values=values, width=22) if values else ttk_module.Entry(tab, textvariable=variable, width=24)
                widget.grid(row=row, column=column + 1, sticky="w", padx=8, pady=6)
            final_field_row = 3 + (len(fields) - 1) // 2
            ttk_module.Checkbutton(tab, text="CUDA mixed precision (AMP)", variable=self.train_amp).grid(row=final_field_row + 1, column=0, columnspan=2, sticky="w", padx=8, pady=8)
            ttk_module.Button(tab, text="Start training", style="Run.TButton", command=lambda: self._guard(messagebox_module, lambda: gui_commands.train(self.train_manifest.get(), self.train_output.get(), self.train_model.get(), self.train_epochs.get(), self.train_batch.get(), self.train_size.get(), self.train_lr.get(), self.train_val.get(), self.train_seed.get(), self.train_device.get(), self.train_workers.get(), self.train_amp.get(), self.train_loss.get(), self.train_recall_target.get(), self.train_recall_weight.get(), self.train_selection_metric.get()))).grid(row=final_field_row + 2, column=2, columnspan=2, sticky="e", padx=8, pady=12)

        def _inference_tab(self, tk_module, ttk_module, filedialog_module, messagebox_module):
            tab = ttk_module.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text="Inference")
            ttk_module.Label(tab, text="Layer 2 — classify each accepted glove crop", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
            self.infer_checkpoint, self.infer_device = tk_module.StringVar(), tk_module.StringVar(value="auto")
            self.infer_decision_class = tk_module.StringVar(value="argmax")
            self.infer_decision_threshold = tk_module.DoubleVar(value=0.5)
            shared = ttk_module.LabelFrame(tab, text="Chirality classifier (separate from Layer 1 detector)", style="Section.TLabelframe", padding=8)
            shared.pack(fill="x", pady=5)
            self._path_row(shared, ttk_module, filedialog_module, 0, "Classifier checkpoint", self.infer_checkpoint, "model")
            ttk_module.Label(shared, text="Device").grid(row=1, column=0, sticky="w", padx=8, pady=5)
            ttk_module.Combobox(shared, textvariable=self.infer_device, values=("auto", "cpu", "cuda", "cuda:0", "cuda:1"), width=18).grid(row=1, column=1, sticky="w", padx=8, pady=5)
            ttk_module.Label(shared, text="Recall-priority class").grid(row=2, column=0, sticky="w", padx=8, pady=5)
            ttk_module.Combobox(shared, textvariable=self.infer_decision_class, values=("argmax", "left", "right"), state="readonly", width=18).grid(row=2, column=1, sticky="w", padx=8, pady=5)
            self._entry(shared, ttk_module, 3, "Class probability threshold", self.infer_decision_threshold, width=20)
            ttk_module.Label(shared, text="For higher right-glove recall, select right and lower the threshold below 0.5; validate the resulting false alarms on held-out sessions.", wraplength=760, foreground="#7a4e20").grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=5)

            video = ttk_module.LabelFrame(tab, text="Full video inference", style="Section.TLabelframe", padding=8)
            video.pack(fill="x", pady=5)
            self.infer_video_path, self.infer_video_output = tk_module.StringVar(), tk_module.StringVar()
            self._path_row(video, ttk_module, filedialog_module, 0, "Video", self.infer_video_path, "video")
            self._path_row(video, ttk_module, filedialog_module, 1, "Output directory", self.infer_video_output, "directory")
            self._config_row(video, ttk_module, filedialog_module, 2)
            ttk_module.Button(video, text="Run video inference", style="Run.TButton", command=lambda: self._guard(messagebox_module, lambda: gui_commands.infer_video(self.infer_video_path.get(), self.infer_checkpoint.get(), self.infer_video_output.get(), self.config_path.get(), self.infer_device.get(), self.infer_decision_class.get(), self.infer_decision_threshold.get()))).grid(row=3, column=1, sticky="e", padx=8, pady=8)

            live = ttk_module.LabelFrame(tab, text="Real-time event inference", style="Section.TLabelframe", padding=8)
            live.pack(fill="x", pady=5)
            self.live_source = tk_module.StringVar(value="0")
            self.live_output = tk_module.StringVar()
            self.live_amp = tk_module.BooleanVar(value=False)
            self._entry(live, ttk_module, 0, "Camera index / stream source", self.live_source)
            self._path_row(live, ttk_module, filedialog_module, 1, "JSONL event output", self.live_output, "save-jsonl")
            ttk_module.Checkbutton(live, text="Classifier CUDA AMP", variable=self.live_amp).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=5)
            live_buttons = ttk_module.Frame(live)
            live_buttons.grid(row=3, column=1, sticky="e", padx=8, pady=8)
            ttk_module.Button(live_buttons, text="Start", style="Run.TButton", command=lambda: self._guard(messagebox_module, lambda: gui_commands.infer_live(self.live_source.get(), self.infer_checkpoint.get(), self.live_output.get(), self.config_path.get(), self.infer_device.get(), self.live_amp.get(), self.infer_decision_class.get(), self.infer_decision_threshold.get()))).pack(side="left", padx=4)
            ttk_module.Button(live_buttons, text="Stop", command=self._stop).pack(side="left", padx=4)

            images = ttk_module.LabelFrame(tab, text="Existing crop inference", style="Section.TLabelframe", padding=8)
            images.pack(fill="x", pady=5)
            self.infer_images_input, self.infer_images_output = tk_module.StringVar(), tk_module.StringVar()
            self._path_row(images, ttk_module, filedialog_module, 0, "Crop/image input", self.infer_images_input, "image-or-directory")
            self._path_row(images, ttk_module, filedialog_module, 1, "Prediction CSV", self.infer_images_output, "save-csv")
            ttk_module.Button(images, text="Classify images", command=lambda: self._guard(messagebox_module, lambda: gui_commands.infer_images(self.infer_images_input.get(), self.infer_checkpoint.get(), self.infer_images_output.get(), self.infer_device.get(), self.infer_decision_class.get(), self.infer_decision_threshold.get()))).grid(row=2, column=1, sticky="e", padx=8, pady=8)

        def _log_panel(self, tk_module, ttk_module, scrolledtext_module):
            tab = ttk_module.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text="Run log")
            controls = ttk_module.Frame(tab)
            controls.pack(fill="x", pady=(0, 6))
            ttk_module.Label(controls, text="Pipeline command output", style="Title.TLabel").pack(side="left")
            ttk_module.Button(controls, text="Clear", command=self._clear_log).pack(side="right", padx=3)
            self.stop_button = ttk_module.Button(
                controls, text="Stop", command=self._stop, state="disabled"
            )
            self.stop_button.pack(side="right", padx=3)
            self.log = scrolledtext_module.ScrolledText(
                tab, wrap="word", font=("Consolas", 9), state="disabled"
            )
            self.log.pack(fill="both", expand=True)

        @staticmethod
        def _parse_box(text: str) -> tuple[float, float, float, float]:
            values = tuple(float(value.strip()) for value in text.split(","))
            if len(values) != 4 or not all(0 <= value <= 1 for value in values):
                raise ValueError("ROI and trigger boxes need four comma-separated values in [0, 1]")
            return values

        def _load_settings(self, messagebox_module, quiet=False):
            try:
                config = ExtractionConfig.from_yaml(self.config_path.get())
                detector, event = config.detector, config.event
                for key in ("backend", "require_full_containment", "trigger_inner_margin_ratio", "color_distance_threshold", "motion_assist", "adaptive_background", "mog_empty_learning_rate", "mog_foreground_learning_rate", "morph_kernel", "min_area_ratio", "max_area_ratio", "yolo_model", "yolo_confidence", "yolo_class_id", "yolo_device", "yolo_half", "yolo_imgsz", "yolo_iou", "yolo_max_det", "yolo_min_box_area_ratio", "yolo_max_box_area_ratio", "yolo_use_masks", "yolo_require_masks", "yolo_crop_to_roi"):
                    value = getattr(detector, key)
                    if key == "yolo_class_id" and value is None:
                        value = 0
                    self.setting_vars[key].set(value)
                self.setting_vars["roi"].set(", ".join(str(value) for value in detector.roi))
                self.setting_vars["trigger_zone"].set(", ".join(str(value) for value in detector.trigger_zone))
                for key in ("min_detected_frames", "reject_multiple_detections", "exit_missing_frames", "cooldown_frames", "crop_padding", "crop_mode", "output_size", "make_square"):
                    self.setting_vars[key].set(getattr(event, key))
                if detector.backend == "yolo" and detector.yolo_model:
                    self.yolo_status.set(
                        "Loaded custom Layer 1 detector. Use Calibration preview before extraction."
                    )
                else:
                    self.yolo_status.set(
                        "Choose the trained glove segmentation checkpoint (usually best.pt)."
                    )
                if not quiet:
                    self._append_log(f"Loaded settings: {self.config_path.get()}\n")
            except (OSError, TypeError, ValueError, yaml.YAMLError, tk.TclError) as exc:
                if not quiet:
                    messagebox_module.showerror("Could not load settings", str(exc))

        def _save_settings(self, messagebox_module):
            try:
                path = self.config_path.get().strip()
                if not path:
                    raise ValueError("Choose a config file path")
                config = ExtractionConfig.from_yaml(path) if Path(path).exists() else ExtractionConfig()
                detector, event = config.detector, config.event
                detector.backend = self.setting_vars["backend"].get()
                detector.roi = self._parse_box(self.setting_vars["roi"].get())
                detector.trigger_zone = self._parse_box(self.setting_vars["trigger_zone"].get())
                for key in ("require_full_containment", "trigger_inner_margin_ratio", "color_distance_threshold", "motion_assist", "adaptive_background", "mog_empty_learning_rate", "mog_foreground_learning_rate", "morph_kernel", "min_area_ratio", "max_area_ratio", "yolo_model", "yolo_confidence", "yolo_class_id", "yolo_device", "yolo_half", "yolo_imgsz", "yolo_iou", "yolo_max_det", "yolo_min_box_area_ratio", "yolo_max_box_area_ratio", "yolo_use_masks", "yolo_require_masks", "yolo_crop_to_roi"):
                    setattr(detector, key, self.setting_vars[key].get())
                for key in ("min_detected_frames", "reject_multiple_detections", "exit_missing_frames", "cooldown_frames", "crop_padding", "crop_mode", "output_size", "make_square"):
                    setattr(event, key, self.setting_vars[key].get())
                detector.validate()
                event.validate()
                config.to_yaml(path)
                self._append_log(f"Saved settings: {path}\n")
            except (OSError, TypeError, ValueError, yaml.YAMLError, tk.TclError) as exc:
                messagebox_module.showerror("Could not save settings", str(exc))

        def _guard(self, messagebox_module, builder):
            try:
                self._run(builder())
            except (OSError, RuntimeError, TypeError, ValueError, tk.TclError) as exc:
                messagebox_module.showerror("Cannot start", str(exc))

        def _run(self, command: list[str]):
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("Another pipeline command is already running")
            self._append_log("\n$ " + subprocess.list2cmdline(command) + "\n")
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=flags,
            )
            self.stop_button.configure(state="normal")

            def collect():
                assert self.process is not None and self.process.stdout is not None
                for line in self.process.stdout:
                    self.messages.put(("line", line))
                code = self.process.wait()
                self.messages.put(("done", code))

            threading.Thread(target=collect, daemon=True).start()

        def _stop(self):
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self._append_log("Stop requested.\n")

        def _drain_messages(self):
            try:
                while True:
                    kind, value = self.messages.get_nowait()
                    if kind == "line":
                        self._append_log(str(value))
                    else:
                        self._append_log(f"Process finished with exit code {value}.\n")
                        self.stop_button.configure(state="disabled")
            except queue.Empty:
                pass
            self.root.after(100, self._drain_messages)

        def _append_log(self, text: str):
            self.log.configure(state="normal")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")

        def _clear_log(self):
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.configure(state="disabled")

    root = tk.Tk()
    App(root)
    if args.smoke_test:
        root.update_idletasks()
        root.update()
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
