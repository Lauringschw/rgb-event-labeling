import numpy as np
import os
from sklearn.model_selection import train_test_split

class GestureDataset:
    def __init__(self, base_folder='/home/lau/Documents/test_1'):
        self.base_folder = base_folder
        self.gestures = ['rock', 'paper', 'scissor']
        self.landmarks = ['t_initial', 't_early', 't_mid', 't_late']
        self.windows = ['20ms', '30ms', '50ms']
        
        # gesture to label mapping
        self.gesture_to_label = {'rock': 0, 'paper': 1, 'scissor': 2}
        self.label_to_gesture = {0: 'rock', 1: 'paper', 2: 'scissor'}
        
    def load_all_samples(self):
        """load all samples organized by landmark and window"""
        dataset = {}
        
        # initialize empty lists for each sample type
        for landmark in self.landmarks:
            for window in self.windows:
                sample_type = f'{landmark}_{window}'
                dataset[sample_type] = {'data': [], 'labels': [], 'metadata': []}
        
        # load samples from all recordings
        missing_count = 0
        for gesture in self.gestures:
            for i in range(1, 21):
                folder_name = f'{gesture[0]}_{i}'  # r_1, p_1, s_1, etc.
                sample_path = os.path.join(self.base_folder, gesture, folder_name, 'event_samples.npy')
                
                if not os.path.exists(sample_path):
                    print(f'⚠ missing: {gesture}/{folder_name}')
                    missing_count += 1
                    continue
                
                # load samples
                samples = np.load(sample_path, allow_pickle=True).item()
                label = self.gesture_to_label[gesture]
                
                # organize by sample type
                for sample_name, event_frame in samples.items():
                    if sample_name in dataset:
                        dataset[sample_name]['data'].append(event_frame)
                        dataset[sample_name]['labels'].append(label)
                        dataset[sample_name]['metadata'].append({
                            'gesture': gesture,
                            'recording': folder_name,
                            'sample_type': sample_name
                        })
        
        print(f'\n✓ loaded samples from {60 - missing_count}/60 recordings')
        
        # convert to numpy arrays
        for sample_type in dataset:
            dataset[sample_type]['data'] = np.array(dataset[sample_type]['data'])
            dataset[sample_type]['labels'] = np.array(dataset[sample_type]['labels'])
        
        return dataset
    
    def get_landmark_dataset(self, dataset, landmark, test_size=0.2, val_size=0.1):
        """get train/val/test split for a specific landmark (all window sizes)"""
        
        # combine all window sizes for this landmark
        all_data = []
        all_labels = []
        
        for window in self.windows:
            sample_type = f'{landmark}_{window}'
            all_data.append(dataset[sample_type]['data'])
            all_labels.append(dataset[sample_type]['labels'])
        
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # split: 70% train, 10% val, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        print(f'\n{landmark} dataset:')
        print(f'  train: {len(X_train)} samples')
        print(f'  val: {len(X_val)} samples')
        print(f'  test: {len(X_test)} samples')
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def get_window_dataset(self, dataset, landmark, window, test_size=0.2, val_size=0.1):
        """get train/val/test split for specific landmark + window combination"""
        
        sample_type = f'{landmark}_{window}'
        X = dataset[sample_type]['data']
        y = dataset[sample_type]['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        print(f'\n{sample_type} dataset:')
        print(f'  train: {len(X_train)} samples')
        print(f'  val: {len(X_val)} samples')
        print(f'  test: {len(X_test)} samples')
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }


if __name__ == '__main__':
    loader = GestureDataset()
    
    # load all samples
    print('loading all samples...')
    dataset = loader.load_all_samples()
    
    # show stats
    print('\n=== dataset statistics ===')
    for sample_type in dataset:
        n_samples = len(dataset[sample_type]['data'])
        print(f'{sample_type}: {n_samples} samples')
    
    # example: get t_initial dataset (all windows)
    t_initial_data = loader.get_landmark_dataset(dataset, 't_initial')
    
    # example: get specific window dataset
    t_initial_20ms_data = loader.get_window_dataset(dataset, 't_initial', '20ms')