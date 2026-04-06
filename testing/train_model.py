# train_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset_loader import GestureDataset

class EventFrameDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # add channel dim: (N, 1, H, W)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # calculate flattened size: 720x1280 -> 90x160 after 3 maxpools
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 90 * 160, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f'epoch {epoch+1}/{epochs}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%')
    
    return model

def test_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    test_acc = 100. * correct / total
    print(f'\ntest accuracy: {test_acc:.2f}%')
    return test_acc

if __name__ == '__main__':
    # check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    
    # load dataset
    loader = GestureDataset()
    dataset = loader.load_all_samples()
    
    # train separate model for each landmark (RQ2)
    results = {}
    
    for landmark in ['t_initial', 't_early', 't_mid', 't_late']:
        print(f'\n{"="*60}')
        print(f'training model for {landmark}')
        print(f'{"="*60}')
        
        # get data
        data = loader.get_landmark_dataset(dataset, landmark)
        
        # create datasets
        train_dataset = EventFrameDataset(data['X_train'], data['y_train'])
        val_dataset = EventFrameDataset(data['X_val'], data['y_val'])
        test_dataset = EventFrameDataset(data['X_test'], data['y_test'])
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)
        
        # train
        model = SimpleCNN(num_classes=3)
        model = train_model(model, train_loader, val_loader, epochs=50, device=device)
        
        # test
        test_acc = test_model(model, test_loader, device=device)
        results[landmark] = test_acc
        
        # save model
        torch.save(model.state_dict(), f'model_{landmark}.pth')
    
    #  summary
    print(f'\n{"="*60}')
    print('RESULTS SUMMARY (RQ2: temporal landmark comparison)')
    print(f'{"="*60}')
    for landmark, acc in results.items():
        print(f'{landmark}: {acc:.2f}% test accuracy')