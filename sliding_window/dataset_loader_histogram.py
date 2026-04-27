from pathlib import Path
import numpy as np
import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR = Path(os.getenv("SLIDING_DIR"))

class HistogramDataset:
    
    def __init__(self):
        self.gestures = ['rock', 'paper', 'scissor']
        self.gesture_to_label = {'rock': 0, 'paper': 1, 'scissor': 2}
        self.label_to_gesture = {0: 'rock', 1: 'paper', 2: 'scissor'}
    
    def load_samples(self):
        """Load consolidated histogram dataset"""
        data_path = SLIDING_DIR / "histogram_data.npy"
        labels_path = SLIDING_DIR / "histogram_labels.npy"
        
        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Dataset not found. Run extract_samples_histogram.py first.\n"
                f"Expected:\n  - {data_path}\n  - {labels_path}"
            )
        
        data = np.load(data_path)
        labels = np.load(labels_path)
        
        print(f'- loaded {len(data)} samples from {SLIDING_DIR}')
        print(f'- data shape: {data.shape}')
        
        # Print class distribution
        for gesture_idx in range(3):
            count = np.sum(labels == gesture_idx)
            gesture_name = self.label_to_gesture[gesture_idx]
            print(f'  {gesture_name}: {count} samples')
        
        return {
            'data': data,
            'labels': labels
        }
    
    def get_split(self, dataset, test_size=0.2, val_size=0.1):
        """Get train/val/test split"""
        X = dataset['data']
        y = dataset['labels']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=123, stratify=y_temp
        )
        
        print(f'\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }