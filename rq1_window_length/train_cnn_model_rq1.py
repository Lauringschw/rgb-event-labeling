import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

# Add rqs_shared to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'rqs_shared'))

from dataset_loader import GestureDataset

# ===== Simple CNN for small dataset =====
class SimpleCNN(nn.Module):
    """Lightweight 3-layer CNN for 1200 samples
    Much simpler than ResNet to reduce overfitting
    """
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: input 1x720x1280 -> 32x360x640
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x180x320
            
            # Layer 2: 32x180x320 -> 64x90x160
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64x45x80
            
            # Layer 3: 64x45x80 -> 128x23x40
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 128x11x20
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 11 * 20, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def train_window_model(window, split, epochs=50, batch_size=16, lr=0.001):
    """Train model for specific window length"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining on {device}')
    
    # prepare data
    X_train = split['X_train'][:, np.newaxis, :, :]  # add channel dim
    X_val = split['X_val'][:, np.newaxis, :, :]
    X_test = split['X_test'][:, np.newaxis, :, :]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(split['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(split['y_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(split['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # model, loss, optimizer
    model = SimpleCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # early stopping
    patience = 15
    patience_counter = 0
    best_val_acc = 0
    best_model_state = None
    
    print(f'\n=== Training {window} model ===')
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        
        # learning rate scheduling
        scheduler.step(val_acc)
        
        # track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc*100:.2f}%')
        
        # early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break
    
    # load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f'\n{window} Results:')
    print(f'  Best val accuracy: {best_val_acc*100:.2f}%')
    print(f'  Test accuracy: {test_acc*100:.2f}%')
    
    # confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f'\nConfusion Matrix:')
    print(cm)
    
    # classification report
    print(f'\nClassification Report:')
    print(classification_report(test_labels, test_preds, 
                                target_names=['rock', 'paper', 'scissor']))
    
    return {
        'window': window,
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'confusion_matrix': cm,
        'model_state': best_model_state
    }

if __name__ == '__main__':
    # load dataset
    loader = GestureDataset()
    print('Loading RQ1 samples...')
    rq1_data = loader.load_rq1_samples()
    
    results = {}
    
    # train model for each window length
    for window in ['100ms', '150ms', '200ms']:
        split = loader.get_rq1_split(rq1_data, window)
        result = train_window_model(window, split, epochs=50, batch_size=16)
        results[window] = result
        
        # save model
        model_path = Path(os.getenv('MODEL_DIR')) / f'rq1_{window}.pth'
        torch.save(result['model_state'], model_path)
        print(f'\nSaved model to {model_path}')
    
    # summary
    print('\n' + '='*60)
    print('RQ1 RESULTS SUMMARY: Window Length Comparison')
    print('='*60)
    for window in ['100ms', '150ms', '200ms']:
        print(f'{window:6s}: test_acc={results[window]["test_accuracy"]*100:.2f}%')
    
    # save results
    results_path = Path(os.getenv('RESULTS_DIR')) / 'rq1_results.npy'
    np.save(results_path, results)
    print(f'\nResults saved to {results_path}')