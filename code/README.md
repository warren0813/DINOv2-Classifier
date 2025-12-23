# DINOv2 Real/Fake Classifier

This project fine-tunes a DINOv2 backbone on a binary classification task: distinguishing real images from AI-generated (fake) images.

## Paths to adjust
- Training data default: `/mnt/c/Users/warre/workspace/train` (expects `fake/` and `real/` subfolders). Override with `--train-dir <path>`.
- Test data default: `/mnt/c/Users/warre/workspace/test`. Override with `--test-dir <path>`.
- Checkpoint output default: `checkpoints/best.pt` (created automatically). Override with `--checkpoint <path>`.
- Submission output default: `submission.csv`. Override with `--submission <path>`.

## Environment
- Python 3.10+ recommended.
- PyTorch + torchvision required; the script auto-downloads the DINOv2 backbone via `torch.hub` (facebookresearch/dinov2).
- CUDA is used if available; mixed precision is on by default for speed.

## Train
```bash
cd /home/warren/deeplearning/final_project
python team_3_DINOv2.py --mode train \
  --train-dir /mnt/c/Users/warre/workspace/train \
  --val-ratio 0.1 \
  --epochs 5 \
  --batch-size 32 \
  --img-size 256
```
Key outputs:
- Best model: `checkpoints/best.pt`
- Training log: console JSON per epoch
- Metrics history: `checkpoints/history.json`

### Common tweaks
- Memory tight: lower `--img-size` or raise `--accumulation-steps`.
- Longer training: increase `--epochs`.
- Different backbone size: `--backbone dinov2_vitb14` (or other names from the hub repo).

## Evaluation (on held-out validation split)
Validation runs automatically each epoch during training using `--val-ratio`. The best checkpoint is chosen by highest validation accuracy. Results live in `checkpoints/history.json` and console output.

To re-evaluate a saved checkpoint on a specific folder split:
```bash
python - <<'PY'
from pathlib import Path
import torch
from team_3_DINOv2 import DinoV2Classifier, RealFakeDataset, build_transforms, evaluate
from torch.utils.data import DataLoader

ckpt_path = Path('checkpoints/best.pt')
data_dir = Path('/mnt/c/Users/warre/workspace/train')  # replace with your eval folder containing fake/ and real/
img_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, eval_tfms = build_transforms(img_size)
val_ds = RealFakeDataset(data_dir, transform=eval_tfms)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
ckpt = torch.load(ckpt_path, map_location=device)
model = DinoV2Classifier(backbone_name=ckpt.get('cfg', {}).get('backbone', 'dinov2_vits14'))
model.load_state_dict(ckpt['model'], strict=False)
model.to(device)
val_loss, val_acc = evaluate(model, val_loader, device)
print({'val_loss': val_loss, 'val_acc': val_acc})
PY
```

## Predict / submission (test set)
```bash
python team_3_DINOv2.py --mode predict \
  --checkpoint checkpoints/best.pt \
  --test-dir /mnt/c/Users/warre/workspace/test \
  --submission submission.csv \
  --img-size 256
```
Output:
- Submission CSV matching `sample_submission.csv` format with columns `filename,label`; labels are `real`/`fake` using a 0.5 probability threshold for the `real` class.

## File map
- Training/inference script: [team_3_DINOv2.py](team_3_DINOv2.py)
- Sample submission template: [sample_submission.csv](sample_submission.csv)
- Generated assets after training: `checkpoints/best.pt`, `checkpoints/history.json`, and your chosen `submission.csv`.
