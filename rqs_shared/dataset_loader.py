import numpy as np
import os
from sklearn.model_selection import train_test_split

class GestureDataset:
    def __init__(self, base_folder='/home/lau/Documents/test_2'):
        self.base_folder = base_folder
        self.gestures = ['rock', 'paper', 'scissor']
        
        # gesture to label mapping
        self.gesture_to_label = {'rock': 0, 'paper': 1, 'scissor': 2}
        self.label_to_gesture = {0: 'rock', 1: 'paper', 2: 'scissor'}
    
    # ======== RQ1: Window Length ========
    def load_rq1_samples(self):
        """Load RQ1 samples: t_initial with different window lengths"""
        dataset = {
            '20ms': {'data': [], 'labels': []},
            '30ms': {'data': [], 'labels': []},
            '50ms': {'data': [], 'labels': []}
        }
        
        missing_count = 0
        for gesture in self.gestures:
            for i in range(1, 21):
                folder_name = f'{gesture[0]}_{i}'
                sample_path = os.path.join(self.base_folder, gesture, folder_name, 'event_samples_rq1.npy')
                
                if not os.path.exists(sample_path):
                    print(f'⚠ missing: {gesture}/{folder_name}')
                    missing_count += 1
                    continue
                
                samples = np.load(sample_path, allow_pickle=True).item()
                label = self.gesture_to_label[gesture]
                
                for window in ['20ms', '30ms', '50ms']:
                    dataset[window]['data'].append(samples[window])
                    dataset[window]['labels'].append(label)
        
        print(f'✓ loaded RQ1 samples from {60 - missing_count}/60 recordings')
        
        # convert to arrays
        for window in dataset:
            dataset[window]['data'] = np.array(dataset[window]['data'])
            dataset[window]['labels'] = np.array(dataset[window]['labels'])
        
        return dataset
    
    def get_rq1_split(self, dataset, window, test_size=0.2, val_size=0.1):
        """Get train/val/test split for RQ1"""
        X = dataset[window]['data']
        y = dataset[window]['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        print(f'{window}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    # ======== RQ2: Temporal Landmark ========
    def load_rq2_samples(self):
        """Load RQ2 samples: different landmarks with 50ms window"""
        dataset = {
            't_initial': {'data': [], 'labels': []},
            't_early': {'data': [], 'labels': []},
            't_mid': {'data': [], 'labels': []},
            't_late': {'data': [], 'labels': []}
        }
        
        missing_count = 0
        for gesture in self.gestures:
            for i in range(1, 21):
                folder_name = f'{gesture[0]}_{i}'
                sample_path = os.path.join(self.base_folder, gesture, folder_name, 'event_samples_rq2.npy')
                
                if not os.path.exists(sample_path):
                    print(f'⚠ missing: {gesture}/{folder_name}')
                    missing_count += 1
                    continue
                
                samples = np.load(sample_path, allow_pickle=True).item()
                label = self.gesture_to_label[gesture]
                
                for landmark in ['t_initial', 't_early', 't_mid', 't_late']:
                    dataset[landmark]['data'].append(samples[landmark])
                    dataset[landmark]['labels'].append(label)
        
        print(f'✓ loaded RQ2 samples from {60 - missing_count}/60 recordings')
        
        # convert to arrays
        for landmark in dataset:
            dataset[landmark]['data'] = np.array(dataset[landmark]['data'])
            dataset[landmark]['labels'] = np.array(dataset[landmark]['labels'])
        
        return dataset
    
    def get_rq2_split(self, dataset, landmark, test_size=0.2, val_size=0.1):
        """Get train/val/test split for RQ2"""
        X = dataset[landmark]['data']
        y = dataset[landmark]['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        print(f'{landmark}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    # ======== RQ3: Event Representation ========
    def load_rq3_samples(self):
        """Load RQ3 samples: different representations at t_initial 50ms"""
        dataset = {
            'histogram': {'data': [], 'labels': []},
            'voxel_grid': {'data': [], 'labels': []},
            'time_surface': {'data': [], 'labels': []}
        }
        
        missing_count = 0
        for gesture in self.gestures:
            for i in range(1, 21):
                folder_name = f'{gesture[0]}_{i}'
                sample_path = os.path.join(self.base_folder, gesture, folder_name, 'event_samples_rq3.npy')
                
                if not os.path.exists(sample_path):
                    print(f'⚠ missing: {gesture}/{folder_name}')
                    missing_count += 1
                    continue
                
                samples = np.load(sample_path, allow_pickle=True).item()
                label = self.gesture_to_label[gesture]
                
                for rep in ['histogram', 'voxel_grid', 'time_surface']:
                    dataset[rep]['data'].append(samples[rep])
                    dataset[rep]['labels'].append(label)
        
        print(f'✓ loaded RQ3 samples from {60 - missing_count}/60 recordings')
        
        # convert to arrays
        for rep in dataset:
            dataset[rep]['data'] = np.array(dataset[rep]['data'])
            dataset[rep]['labels'] = np.array(dataset[rep]['labels'])
        
        return dataset
    
    def get_rq3_split(self, dataset, representation, test_size=0.2, val_size=0.1):
        """Get train/val/test split for RQ3"""
        X = dataset[representation]['data']
        y = dataset[representation]['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        print(f'{representation}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
