# rq3_representation/train_model_rq3.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.path.append('..')
from dataset_loader import GestureDataset

class GestureCNN2D(nn.Module):
    """2D CNN for histogram and time_surface (720 x 1280 inputs)"""
    def __init__(self):
        super(GestureCNN2D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 11 * 20, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class GestureCNN3D(nn.Module):
    """3D CNN for voxel_grid (5 x 720 x 1280 inputs)"""
    def __init__(self):
        super(GestureCNN3D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # input: 1 x 5 x 720 x 1280
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # -> 32 x 5 x 180 x 320
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # -> 64 x 5 x 45 x 80
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # -> 128 x 2 x 11 x 20
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 11 * 20, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
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

def train_representation_model(representation, split, epochs=50, batch_size=16, lr=0.001):
    """Train model for specific event representation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining on {device}')
    
    # prepare data based on representation type
    if representation == 'voxel_grid':
        # 3D input: add channel dim -> (N, 1, 5, 720, 1280)
        X_train = split['X_train'][:, np.newaxis, :, :, :]
        X_val = split['X_val'][:, np.newaxis, :, :, :]
        X_test = split['X_test'][:, np.newaxis, :, :, :]
        model = GestureCNN3D().to(device)
    else:
        # 2D input: add channel dim -> (N, 1, 720, 1280)
        X_train = split['X_train'][:, np.newaxis, :, :]
        X_val = split['X_val'][:, np.newaxis, :, :]
        X_test = split['X_test'][:, np.newaxis, :, :]
        model = GestureCNN2D().to(device)
    
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
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # training loop
    best_val_acc = 0
    best_model_state = None
    
    print(f'\n=== Training {representation} model ===')
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%')
    
    # load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f'\n{representation} Results:')
    print(f'  Best val accuracy: {best_val_acc:.2f}%')
    print(f'  Test accuracy: {test_acc:.2f}%')
    
    # confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f'\nConfusion Matrix:')
    print(cm)
    
    # classification report
    print(f'\nClassification Report:')
    print(classification_report(test_labels, test_preds, 
                                target_names=['rock', 'paper', 'scissor']))
    
    return {
        'representation': representation,
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'confusion_matrix': cm,
        'model_state': best_model_state
    }

if __name__ == '__main__':
    # load dataset
    loader = GestureDataset()
    print('Loading RQ3 samples...')
    rq3_data = loader.load_rq3_samples()
    
    results = {}
    
    # train model for each representation
    for representation in ['histogram', 'voxel_grid', 'time_surface']:
        split = loader.get_rq3_split(rq3_data, representation)
        result = train_representation_model(representation, split, epochs=50, batch_size=16)
        results[representation] = result
        
        # save model
        torch.save(result['model_state'], f'model_{representation}.pth')
        print(f'\nSaved model to model_{representation}.pth')
    
    # summary
    print('\n' + '='*60)
    print('RQ3 RESULTS SUMMARY: Event Representation Comparison')
    print('='*60)
    for rep in ['histogram', 'voxel_grid', 'time_surface']:
        print(f'{rep:<15}: test_acc={results[rep]["test_accuracy"]:.2f}%')
    
    # save results
    np.save('rq3_results.npy', results)
    print('\nResults saved to rq3_results.npy')