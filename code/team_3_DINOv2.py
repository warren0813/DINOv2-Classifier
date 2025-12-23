"""Train and run a DINOv2-based classifier for real vs fake images.

Usage examples
--------------
Train:
	python team_3_DINOv2.py --mode train \
		--train-dir /mnt/c/Users/warre/workspace/train \
		--val-ratio 0.1 \
		--epochs 5 \
		--batch-size 32

Predict / submission:
	python team_3_DINOv2.py --mode predict \
		--checkpoint checkpoints/best.pt \
		--test-dir /mnt/c/Users/warre/workspace/test \
		--submission sample_submission.csv

Notes
-----
- The script auto-downloads the DINOv2 backbone via torch.hub (facebookresearch/dinov2).
- Mixed precision is enabled when CUDA is available for faster training.
- Adjust image size with --img-size if memory is tight.
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets.folder import default_loader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _maybe_install(package: str) -> None:
	"""Install a package with pip if it is missing."""

	try:
		__import__(package)
	except ImportError:
		subprocess.check_call(["pip", "install", package])


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True


def num_workers_hint() -> int:
	try:
		import psutil

		return max(2, min(8, psutil.cpu_count(logical=True) // 2))
	except Exception:
		return 4


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


class RealFakeDataset(Dataset):
	def __init__(self, root: str | Path, transform=None):
		self.root = Path(root)
		self.transform = transform
		self.samples = []

		fake_dir = self.root / "fake"
		real_dir = self.root / "real"

		for label, directory in enumerate([fake_dir, real_dir]):
			for img_path in sorted(directory.glob("**/*")):
				if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
					self.samples.append((img_path, label))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		path, label = self.samples[idx]
		image = default_loader(path)
		if self.transform:
			image = self.transform(image)
		return image, label


class TestImageDataset(Dataset):
	def __init__(self, root: str | Path, transform=None):
		self.root = Path(root)
		self.transform = transform
		self.samples = []
		for img_path in sorted(self.root.glob("**/*")):
			if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
				self.samples.append(img_path)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
		path = self.samples[idx]
		image = default_loader(path)
		if self.transform:
			image = self.transform(image)
		return image, path.name


def build_transforms(img_size: int = 256) -> tuple[transforms.Compose, transforms.Compose]:
	train_tfms = transforms.Compose(
		[
			transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandAugment(num_ops=2, magnitude=9),
			transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	eval_tfms = transforms.Compose(
		[
			transforms.Resize(int(img_size * 1.15)),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return train_tfms, eval_tfms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DinoV2Classifier(nn.Module):
	def __init__(self, backbone_name: str = "dinov2_vits14", num_classes: int = 2, drop: float = 0.2):
		super().__init__()

		# Lazily import dinov2 via torch.hub to avoid heavy dependencies if unused
		_maybe_install("torchvision")
		self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
		for p in self.backbone.parameters():
			p.requires_grad = False  # freeze to start; we only fine-tune the head by default

		embed_dim = self.backbone.embed_dim
		self.head = nn.Sequential(
			nn.LayerNorm(embed_dim),
			nn.Dropout(drop),
			nn.Linear(embed_dim, embed_dim // 2),
			nn.GELU(),
			nn.Dropout(drop),
			nn.Linear(embed_dim // 2, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feats = self.backbone(x)
		if isinstance(feats, (list, tuple)):
			feats = feats[0]
		# The backbone returns a feature map; pool tokens if necessary
		if feats.dim() == 3:
			feats = feats.mean(dim=1)
		return self.head(feats)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
	train_dir: Path
	val_ratio: float = 0.1
	batch_size: int = 32
	epochs: int = 5
	lr: float = 2e-4
	weight_decay: float = 1e-4
	img_size: int = 256
	backbone: str = "dinov2_vits14"
	num_workers: int = num_workers_hint()
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	accumulation_steps: int = 1
	label_smoothing: float = 0.05
	checkpoint_dir: Path = Path("checkpoints")


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler,
	device: torch.device,
	accumulation_steps: int = 1,
	label_smoothing: float = 0.0,
) -> Tuple[float, float]:
	model.train()
	total_loss, total_correct, total_samples = 0.0, 0, 0
	criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

	for step, (images, targets) in enumerate(loader):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
			logits = model(images)
			loss = criterion(logits, targets) / accumulation_steps

		scaler.scale(loss).backward()
		if (step + 1) % accumulation_steps == 0:
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)

		with torch.no_grad():
			preds = logits.argmax(dim=1)
			total_correct += (preds == targets).sum().item()
			total_samples += targets.size(0)
			total_loss += loss.item() * accumulation_steps

	return total_loss / max(1, len(loader)), total_correct / max(1, total_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	total_loss, total_correct, total_samples = 0.0, 0, 0
	criterion = nn.CrossEntropyLoss()
	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits = model(images)
		loss = criterion(logits, targets)
		preds = logits.argmax(dim=1)
		total_correct += (preds == targets).sum().item()
		total_samples += targets.size(0)
		total_loss += loss.item()

	return total_loss / max(1, len(loader)), total_correct / max(1, total_samples)


def train_model(cfg: TrainConfig) -> Path:
	set_seed()
	cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	train_tfms, eval_tfms = build_transforms(cfg.img_size)
	full_dataset = RealFakeDataset(cfg.train_dir, transform=train_tfms)

	val_size = int(len(full_dataset) * cfg.val_ratio)
	train_size = len(full_dataset) - val_size
	train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

	# For validation, reuse validation transforms
	val_ds.dataset.transform = eval_tfms

	train_loader = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=True,
		drop_last=True,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)

	device = torch.device(cfg.device)
	model = DinoV2Classifier(backbone_name=cfg.backbone).to(device)

	optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

	best_acc = 0.0
	best_path = cfg.checkpoint_dir / "best.pt"
	history = []

	for epoch in range(1, cfg.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, optimizer, scaler, device, accumulation_steps=cfg.accumulation_steps, label_smoothing=cfg.label_smoothing
		)
		val_loss, val_acc = evaluate(model, val_loader, device)
		history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

		# Save best checkpoint
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

		print(
			json.dumps(
				{"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4), "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)}
			)
		)

	with open(cfg.checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)

	return best_path


# ---------------------------------------------------------------------------
# Inference and submission
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> list[Tuple[str, float]]:
	model.eval()
	outputs = []
	for images, names in loader:
		images = images.to(device, non_blocking=True)
		logits = model(images)
		probs = F.softmax(logits, dim=1)[:, 1]  # probability of real class
		outputs.extend(zip(names, probs.cpu().tolist()))
	return outputs


def run_submission(checkpoint: Path, test_dir: Path, output_csv: Path, img_size: int = 256, backbone: str | None = None) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	eval_tfms = build_transforms(img_size)[1]
	test_ds = TestImageDataset(test_dir, transform=eval_tfms)
	test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers_hint(), pin_memory=True)

	ckpt = torch.load(checkpoint, map_location=device)
	model = DinoV2Classifier(backbone_name=backbone or ckpt.get("cfg", {}).get("backbone", "dinov2_vits14"))
	model.load_state_dict(ckpt["model"], strict=False)
	model.to(device)

	outputs = predict(model, test_loader, device)

	# Write submission: label should be real/fake; threshold 0.5
	with open(output_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["filename", "label"])
		for name, prob_real in outputs:
			label = "real" if prob_real >= 0.5 else "fake"
			# Drop extension to match sample if needed
			filename = Path(name).stem if name.find(".") != -1 else name
			writer.writerow([filename, label])


def default_submission_path() -> Path:
	"""Place submission next to sample_submission.csv with a timestamped name."""
	script_dir = Path(__file__).resolve().parent
	sample_csv = script_dir.parent / "sample_submission.csv"
	target_dir = sample_csv.parent if sample_csv.exists() else script_dir
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return target_dir / f"{timestamp}_team_3_submission.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and run DINOv2 real/fake classifier")
	parser.add_argument("--mode", choices=["train", "predict"], required=True)
	parser.add_argument("--train-dir", type=Path, default=Path("/mnt/c/Users/warre/workspace/train"))
	parser.add_argument("--test-dir", type=Path, default=Path("/mnt/c/Users/warre/workspace/test"))
	parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
	parser.add_argument("--submission", type=Path, default=None)
	parser.add_argument("--val-ratio", type=float, default=0.1)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--backbone", type=str, default="dinov2_vits14")
	parser.add_argument("--accumulation-steps", type=int, default=1)
	parser.add_argument("--label-smoothing", type=float, default=0.05)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.mode == "train":
		cfg = TrainConfig(
			train_dir=args.train_dir,
			val_ratio=args.val_ratio,
			batch_size=args.batch_size,
			epochs=args.epochs,
			lr=args.lr,
			weight_decay=args.weight_decay,
			img_size=args.img_size,
			backbone=args.backbone,
			accumulation_steps=args.accumulation_steps,
			label_smoothing=args.label_smoothing,
		)
		best_path = train_model(cfg)
		print(f"Best checkpoint saved to {best_path}")
	else:
		submission_path = args.submission or default_submission_path()
		run_submission(args.checkpoint, args.test_dir, submission_path, img_size=args.img_size, backbone=args.backbone)
		print(f"Submission written to {submission_path}")


if __name__ == "__main__":
	main()