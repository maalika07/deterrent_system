"""
Model Comparison Trainer
Trains EfficientNet-B0 and MobileNetV3-Large on the same dataset as ResNet50.
Identical training protocol: frozen backbone, same LR, same epochs, same split seed.

Usage:
    python train_compare.py

Outputs:
    animal_classifier_efficientnet_b0.pth
    animal_classifier_mobilenet_v3.pth
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
DATASET_PATH  = "D:\Animals_dataset\Animals_dataset"
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 0.001
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TRANSFORMS ────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── FOLDER → CLEAN CLASS NAME MAP ────────────────────────────────
FOLDER_TO_CLASS = {
    'Peacock':                'Peacock',
    'bonnet_macaque_dataset': 'bonnet_macaque',
    'chital_dataset':         'chital',
    'dataset':                'no_animal',
    'dataset2':               'no_animal',
    'elephant_dataset':       'elephant',
    'pig_dataset':            'pig',
    'porcupine_dataset':      'porcupine',
    'street_dogs_dataset':    'street_dogs',
    'wild_boar_dataset':      'wild_boar',
}

# ── REMAPPED DATASET ──────────────────────────────────────────────
class RemappedDataset(Dataset):
    """
    Wraps ImageFolder and remaps folder names to clean class names.
    Merges 'dataset' and 'dataset2' into a single 'no_animal' class.
    """
    def __init__(self, root, transform):
        self.base    = datasets.ImageFolder(root, transform=transform)
        raw_classes  = self.base.classes

        # Build sorted unique clean class list
        clean_names      = [FOLDER_TO_CLASS.get(c, c) for c in raw_classes]
        self.classes     = sorted(set(clean_names))
        clean_to_idx     = {c: i for i, c in enumerate(self.classes)}

        # Map old index → new index
        old_to_new = {
            self.base.class_to_idx[raw]: clean_to_idx[FOLDER_TO_CLASS.get(raw, raw)]
            for raw in raw_classes
        }

        # Remap all samples
        self.samples   = [(p, old_to_new[l]) for p, l in self.base.samples]
        self.transform = transform
        self.loader    = self.base.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── TRAIN FUNCTION ────────────────────────────────────────────────
def train_model(model, model_name, save_path, train_loader, val_loader,
                train_size, val_size):
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    t_start      = time.time()

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct      = 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1:02d}/{EPOCHS} [train]",
                                   leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct      += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / train_size
        train_acc  = correct / train_size

        # ── Validate ──
        model.eval()
        val_loss    = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs      = model(inputs)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / val_size
        val_acc  = val_correct / val_size

        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
              + ("  ← best" if is_best else ""))

    elapsed = time.time() - t_start
    print(f"\n  Best val_acc : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"  Training time: {elapsed/60:.1f} min")
    print(f"  Saved        : {save_path}")
    return best_val_acc


# ── MAIN — must be inside if __name__ == '__main__' on Windows ────
if __name__ == '__main__':

    # Build datasets with clean class names
    train_ds = RemappedDataset(DATASET_PATH, train_tf)
    val_ds   = RemappedDataset(DATASET_PATH, val_tf)

    class_names = train_ds.classes
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    # Same 80/20 split with fixed seed — identical to ResNet50 training
    total      = len(train_ds)
    train_size = int(0.8 * total)
    val_size   = total - train_size

    train_subset, _ = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    _, val_subset = torch.utils.data.random_split(
        val_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # num_workers=0 is required on Windows — avoids multiprocessing spawn error
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train: {train_size} | Val: {val_size} | Device: {DEVICE}\n")

    # ── Model 1: EfficientNet-B0 ──────────────────────────────────
    # 5.3M params, strong accuracy/speed tradeoff
    eff = models.efficientnet_b0(weights="IMAGENET1K_V1")
    for p in eff.parameters():
        p.requires_grad = False                          # freeze backbone
    eff.classifier[1] = nn.Linear(
        eff.classifier[1].in_features, n_classes)       # replace head
    for p in eff.classifier.parameters():
        p.requires_grad = True

    acc_eff = train_model(
        eff, "EfficientNet-B0",
        "animal_classifier_efficientnet_b0.pth",
        train_loader, val_loader, train_size, val_size
    )

    # ── Model 2: MobileNetV3-Large ────────────────────────────────
    # 5.4M params, designed for edge/mobile inference
    mob = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    for p in mob.parameters():
        p.requires_grad = False
    mob.classifier[3] = nn.Linear(
        mob.classifier[3].in_features, n_classes)
    for p in mob.classifier.parameters():
        p.requires_grad = True

    acc_mob = train_model(
        mob, "MobileNetV3-Large",
        "animal_classifier_mobilenet_v3.pth",
        train_loader, val_loader, train_size, val_size
    )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  EfficientNet-B0    best val acc: {acc_eff*100:.2f}%")
    print(f"  MobileNetV3-Large  best val acc: {acc_mob*100:.2f}%")
    print(f"\n  Now run: python compare_models.py")