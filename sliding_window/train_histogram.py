import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import os
from dotenv import load_dotenv

from dataset_loader_histogram import HistogramDataset

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR = Path(os.getenv("SLIDING_DIR"))
SLIDING_DIR.mkdir(parents=True, exist_ok=True)

# CNN architecture for 2-channel histogram
class HistogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # After 3 pooling layers: 720/8 = 90, 1280/8 = 160
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 90 * 160, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Output directory: {SLIDING_DIR}\n')
    
    # Load dataset
    dataset = HistogramDataset()
    data = dataset.load_samples()
    split = dataset.get_split(data)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(split['X_train']),
        torch.LongTensor(split['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(split['X_val']),
        torch.LongTensor(split['y_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(split['X_test']),
        torch.LongTensor(split['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = HistogramCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0
    epochs = 50
    
    print('\n' + '='*50)
    print('Training Histogram CNN')
    print('='*50 + '\n')
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%')
        print(f'  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = SLIDING_DIR / 'model_histogram_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Saved best model')
        print()
    
    # Test evaluation
    print('='*50)
    print('Final Evaluation')
    print('='*50)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest: loss={test_loss:.4f}, acc={test_acc:.2f}%')
    
    # Save final metrics
    metrics_path = SLIDING_DIR / 'histogram_training_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f'Best validation accuracy: {best_val_acc:.2f}%\n')
        f.write(f'Test accuracy: {test_acc:.2f}%\n')
        f.write(f'Test loss: {test_loss:.4f}\n')
    print(f'\n✓ Saved metrics to {metrics_path}')