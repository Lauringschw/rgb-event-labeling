import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv
from torchvision import models

from dataset_loader_histogram import HistogramTimeDataset

import random

load_dotenv(Path(__file__).parent.parent.parent / '.env')

SLIDING_DIR_T7_TIME = Path(os.getenv("SLIDING_DIR_T7_TIME"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
SLIDING_DIR_T7_TIME.mkdir(parents=True, exist_ok=True)

USE_RESNET = True  # Toggle: True = ResNet-18, False = Custom CNN

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# == dataset ===================================================================

class MmapGestureDataset(Dataset):
    """Reads samples one at a time from mmap'd array."""
    def __init__(self, data, labels, indices):
        self.data    = data
        self.labels  = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i].copy().astype(np.float32)
        
        # Normalize
        max_val = x.max()
        if max_val > 0:
            x = x / max_val
        
        x = torch.from_numpy(x)
        y = int(self.labels[i])
        return x, y


# == models ====================================================================

class HistogramCNN(nn.Module):
    """Original 3-conv + 2-FC CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 45 * 80, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet18Histogram(nn.Module):
    """ResNet-18 adapted for 2-channel histogram input"""
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        
        # Load ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer: 3 channels -> 2 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            2,  # 2 input channels (ON/OFF polarities)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # If pretrained, initialize new conv1 from original weights
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :2, :, :] = original_conv1.weight[:, :2, :, :]
        
        # Modify final fully connected layer: 1000 classes -> 3 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# == train / eval ==============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += out.argmax(1).eq(y_batch).sum().item()
        total      += y_batch.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item()
            correct    += out.argmax(1).eq(y_batch).sum().item()
            total      += y_batch.size(0)
    return total_loss / len(loader), 100.0 * correct / total


# == main ======================================================================

if __name__ == "__main__":
    seed_everything(SEED)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("TRAINING: Time-based Histogram Model")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Model      : {'ResNet-18' if USE_RESNET else 'Custom CNN'}")
    print(f"Output dir : {SLIDING_DIR_T7_TIME}\n")

    # == load & split ==========================================================
    dataset = HistogramTimeDataset()
    loaded  = dataset.load_samples()
    split   = dataset.get_split(loaded, test_size=0.20, val_size=0.10)

    raw_data   = loaded['data']
    raw_labels = loaded['labels']

    recording_ids = loaded['recording_ids']
    train_mask = np.isin(recording_ids, split['recs_train'])
    val_mask   = np.isin(recording_ids, split['recs_val'])
    test_mask  = np.isin(recording_ids, split['recs_test'])

    train_idx = np.sort(np.where(train_mask)[0])
    val_idx   = np.sort(np.where(val_mask)[0])  # ← ADD np.sort
    test_idx  = np.sort(np.where(test_mask)[0])  # ← ADD np.sort

    print(f"\nIndex counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # == SAVE TEST RECORDING IDS FOR RQ EVALUATION ============================
    test_rec_ids_path = OUTPUT_DIR / "test_recording_ids.npy"
    np.save(test_rec_ids_path, split['recs_test'])
    print(f"Saved test recording IDs: {test_rec_ids_path}\n")

    # == dataloaders ===========================================================
    def make_loader(indices, shuffle=False):
        ds = MmapGestureDataset(raw_data, raw_labels, indices)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return DataLoader(
            ds, 
            batch_size=32, 
            shuffle=shuffle, 
            num_workers=0,
            generator=generator
    )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx)
    test_loader  = make_loader(test_idx)
    
    
    # == debug first batch =====================================================
    print("Debug — checking first batch...")
    X_debug, y_debug = next(iter(train_loader))
    print(f"  Input shape : {X_debug.shape}")
    print(f"  Input min   : {X_debug.min():.4f}  max: {X_debug.max():.4f}")
    print(f"  Labels      : {y_debug.numpy()}")
    print(f"  Non-zero pixels: {(X_debug != 0).float().mean():.4f}")

    # == model / optimizer / loss ==============================================
    if USE_RESNET:
        model = ResNet18Histogram(num_classes=3, pretrained=False).to(device)
    else:
        model = HistogramCNN().to(device)
    
    class_counts = np.array([
        np.sum(raw_labels[train_idx] == 0),
        np.sum(raw_labels[train_idx] == 1),
        np.sum(raw_labels[train_idx] == 2),
    ])
    print(f"\nTrain class counts: rock={class_counts[0]}, paper={class_counts[1]}, scissor={class_counts[2]}")
    
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    # == training loop =========================================================
    MAX_EPOCHS = 50
    PATIENCE   = 10

    best_val_acc      = 0.0
    epochs_no_improve = 0
    model_name        = 'resnet18' if USE_RESNET else 'custom_cnn'
    model_path        = SLIDING_DIR_T7_TIME / f'model_histogram_time_{model_name}_best.pth'

    print("=" * 50)
    print(f"Training {model_name.upper()}")
    print("=" * 50 + "\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{MAX_EPOCHS}")
        print(f"  Train : loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val   : loss={val_loss:.4f}  acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  => New best val acc={best_val_acc:.2f}% — model saved")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        print()

    # == final test evaluation =================================================
    print("=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  : loss={test_loss:.4f}  acc={test_acc:.2f}%")
    print(f"Best val acc: {best_val_acc:.2f}%")

    # == save metrics ==========================================================
    metrics_path = SLIDING_DIR_T7_TIME / f'histogram_time_{model_name}_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Window: 30ms time-based\n")
        f.write(f"Best validation accuracy : {best_val_acc:.2f}%\n")
        f.write(f"Test accuracy            : {test_acc:.2f}%\n")
        f.write(f"Test loss                : {test_loss:.4f}\n")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Test IDs saved to {test_rec_ids_path}")
    
    