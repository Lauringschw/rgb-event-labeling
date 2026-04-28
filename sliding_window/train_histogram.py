import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import os
from dotenv import load_dotenv

from dataset_loader_histogram import HistogramDataset

load_dotenv(Path(__file__).parent.parent / '.env')

# reads the MERGED file written by merge.py
SLIDING_DIR_T7 = Path(os.getenv("SLIDING_DIR_T7"))
SLIDING_DIR_T7.mkdir(parents=True, exist_ok=True)


# == model =====================================================================

class HistogramCNN(nn.Module):
    """
    3-block CNN for 2-channel histogram input (720×1280).
    After 3× MaxPool2d(2): spatial dims -> 90×160.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 90 * 160, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# == train / eval helpers ======================================================

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
    print(f"Output dir : {SLIDING_DIR_T7}\n")

    # == load & split ==========================================================
    # HistogramDataset reads from SLIDING_DIR_T7
    dataset = HistogramDataset()
    data    = dataset.load_samples()
    split   = dataset.get_split(data, test_size=0.20, val_size=0.10)

    # == dataloaders ===========================================================
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=2,
                          pin_memory=(device.type == 'cuda'))

    train_loader = make_loader(split['X_train'], split['y_train'], shuffle=True)
    val_loader   = make_loader(split['X_val'],   split['y_val'])
    test_loader  = make_loader(split['X_test'],  split['y_test'])

    # == model / optimiser / loss ==============================================
    model     = HistogramCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    # == training loop with early stopping ====================================
    MAX_EPOCHS   = 50
    PATIENCE     = 10   # epochs without val-acc improvement --> stop

    best_val_acc      = 0.0
    epochs_no_improve = 0
    model_path        = SLIDING_DIR_T7 / 'model_histogram_best.pth'

    print("=" * 50)
    print("Training Histogram CNN")
    print("=" * 50 + "\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

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
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break
        print()

    # == final test evaluation (load best weights) ============================
    print("=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  : loss={test_loss:.4f}  acc={test_acc:.2f}%")
    print(f"Best val acc: {best_val_acc:.2f}%")

    # == save metrics =========================================================
    metrics_path = SLIDING_DIR_T7 / 'histogram_training_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Best validation accuracy : {best_val_acc:.2f}%\n")
        f.write(f"Test accuracy            : {test_acc:.2f}%\n")
        f.write(f"Test loss                : {test_loss:.4f}\n")
    print(f"\nMetrics saved to {metrics_path}")