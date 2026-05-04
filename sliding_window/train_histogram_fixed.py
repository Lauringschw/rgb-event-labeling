import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

FIXED_DIR = Path(os.getenv("SLIDING_DIR_T7")) / "histogram_fixed"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MS = 50  # must match extract_samples_histogram_fixed.py


# == dataset ===================================================================

class FixedWindowDataset(Dataset):
    """
    Loads full dataset into RAM — feasible since fixed-time dataset is small (~11GB).
    Applies temporal jitter at load time by slightly perturbing the histogram
    values with small Gaussian noise to increase effective sample diversity.
    """
    def __init__(self, data, labels, augment=False):
        # normalize each sample
        self.data    = np.array(data, dtype=np.float32)
        self.labels  = np.array(labels, dtype=np.int64)
        self.augment = augment

        # normalize per sample
        for i in range(len(self.data)):
            max_val = self.data[i].max()
            if max_val > 0:
                self.data[i] /= max_val

        print(f"  Loaded {len(self.data)} samples into RAM")
        print(f"  Shape: {self.data.shape}  dtype: {self.data.dtype}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        if self.augment:
            # small Gaussian noise to increase diversity
            noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)
        return torch.from_numpy(x), int(self.labels[idx])


# == model =====================================================================

class HistogramCNN(nn.Module):
    """Same architecture as thesis baseline for fair comparison."""
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device     : {device}")
    print(f"Output dir : {FIXED_DIR}\n")

    # == load datasets into RAM ================================================
    print("Loading train set into RAM ...")
    train_data   = np.load(FIXED_DIR / "histogram_fixed_train_data.npy")
    train_labels = np.load(FIXED_DIR / "histogram_fixed_train_labels.npy")

    print("Loading val set into RAM ...")
    val_data   = np.load(FIXED_DIR / "histogram_fixed_val_data.npy")
    val_labels = np.load(FIXED_DIR / "histogram_fixed_val_labels.npy")

    train_dataset = FixedWindowDataset(train_data, train_labels, augment=True)
    val_dataset   = FixedWindowDataset(val_data,   val_labels,   augment=False)

    # free raw arrays
    del train_data, val_data

    # class counts for weighted loss
    class_counts = np.array([
        np.sum(train_labels == 0),
        np.sum(train_labels == 1),
        np.sum(train_labels == 2),
    ])
    print(f"\nTrain class counts: rock={class_counts[0]}, paper={class_counts[1]}, scissor={class_counts[2]}")
    del train_labels

    # == dataloaders ===========================================================
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

    # == model / optimiser / loss ==============================================
    model = HistogramCNN().to(device)

    class_weights = torch.FloatTensor(1.0 / class_counts) * len(class_counts)
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer     = optim.Adam(model.parameters(), lr=0.001)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    # == training loop =========================================================
    MAX_EPOCHS = 100   # more epochs since dataset is small
    PATIENCE   = 15    # more patience since dataset is small

    best_val_acc      = 0.0
    epochs_no_improve = 0
    model_path        = FIXED_DIR / 'model_histogram_fixed_best.pth'

    print("=" * 50)
    print(f"Training Histogram CNN (fixed {WINDOW_MS}ms windows)")
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

    print(f"\nBest val acc: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"\nNext step: run evaluate_histogram.py (update model_path to histogram_fixed)")
    print("The test extraction is identical — no need to re-run extract_test_samples_histogram.py")

    # save metrics
    metrics_path = FIXED_DIR / 'histogram_fixed_training_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Training window    : {WINDOW_MS}ms fixed\n")
        f.write(f"Train offsets      : 0-100ms in 10ms steps\n")
        f.write(f"Best val accuracy  : {best_val_acc:.2f}%\n")
    print(f"Metrics saved to {metrics_path}")