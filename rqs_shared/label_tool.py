import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
from pathlib import Path

class GestureLabelingTool:
    def __init__(self, recording_folder):
        self.recording_folder = Path(recording_folder)
        
        # load trigger timestamps from recording folder
        trigger_path = self.recording_folder / 'basler_frame_timestamps.npy'
        self.trigger_times = np.load(trigger_path)
        self.n_frames = len(self.trigger_times)
        
        # load basler files - SORT BY FRAME NUMBER, NOT ALPHABETICALLY
        self.basler_files = sorted(
            [f for f in os.listdir(self.recording_folder) 
             if f.startswith('Basler') and f.endswith('.raw')],
            key=lambda x: int(x.split('__')[1].split('.')[0])  # Extract frame number
        )[:self.n_frames]
        
        if len(self.basler_files) == 0:
            raise FileNotFoundError(f"No Basler .raw files found in {self.recording_folder}")
        
        self.current_frame = 0
        self.go_frame = None
        self.t_initial_frame = None
        
        # Restore previously saved labels first; fall back to metadata only when needed
        self.load_saved_labels_or_metadata()
        
        # setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        self.img_display = None
        self.setup_ui()
        
        # Jump to GO frame if available
        if self.go_frame is not None:
            self.load_frame(self.go_frame)
        else:
            self.load_frame(0)
    
    def load_saved_labels_or_metadata(self):
        labels_path = self.recording_folder / 'labels.npy'
        if labels_path.exists():
            try:
                labels = np.load(labels_path, allow_pickle=True).item()
                self.go_frame = labels.get('go_frame')
                self.t_initial_frame = labels.get('t_initial_frame')
                print(f"✓ Loaded saved labels from {labels_path}")
                print(f"  GO frame: {self.go_frame}")
                print(f"  t_initial frame: {self.t_initial_frame}")
                return
            except Exception as e:
                print(f"⚠ Could not load labels from {labels_path}: {e}")

        self.load_go_from_metadata()

    def load_go_from_metadata(self):
        """Load GO frame from recording metadata"""
        metadata_path = self.recording_folder / 'recording_metadata.npy'
        
        if not metadata_path.exists():
            print("⚠ No recording_metadata.npy found - GO frame not auto-detected")
            return
        
        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
            expected_go_frame = metadata.get('expected_go_frame')
            
            if expected_go_frame is not None:
                # Use expected frame, bounded by actual frame count
                self.go_frame = min(expected_go_frame, self.n_frames - 1)
                print(f"✓ GO frame auto-loaded: frame {self.go_frame}")
                print(f"  (from metadata: {metadata['go_offset_from_start']:.3f}s after start)")
            else:
                print("⚠ No 'expected_go_frame' in metadata")
        except Exception as e:
            print(f"⚠ Could not load GO from metadata: {e}")
        
    def load_frame(self, frame_idx):
        # bounds check
        frame_idx = max(0, min(frame_idx, self.n_frames - 1))
        
        if frame_idx >= len(self.basler_files):
            print(f"Warning: frame {frame_idx} out of range (only {len(self.basler_files)} files)")
            return
        
        frame_path = self.recording_folder / self.basler_files[frame_idx]
        
        try:
            frame = np.fromfile(frame_path, dtype=np.uint8).reshape(1200, 1920)
        except Exception as e:
            print(f"Error loading frame {frame_idx} from {frame_path}: {e}")
            return
        
        if self.img_display is None:
            self.img_display = self.ax.imshow(frame, cmap='gray')
        else:
            self.img_display.set_data(frame)
        
        self.current_frame = frame_idx
        self.update_title()
        self.fig.canvas.draw_idle()
    
    def update_title(self):
        # Calculate time relative to first frame in milliseconds
        time_relative_ms = (self.trigger_times[self.current_frame] - self.trigger_times[0]) / 1000
        
        title = f"{self.recording_folder.name} | Frame {self.current_frame}/{self.n_frames-1} | "
        title += f"Time: {time_relative_ms:.1f}ms"
        
        if self.go_frame is not None:
            title += f" | GO: frame {self.go_frame}"
        if self.t_initial_frame is not None:
            title += f" | t_initial: frame {self.t_initial_frame}"
        
        self.ax.set_title(title)
    
    def setup_ui(self):
        # slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames-1, 
                             valinit=self.go_frame if self.go_frame is not None else 0, 
                             valstep=1)
        self.slider.on_changed(lambda val: self.load_frame(int(val)))
        
        # buttons
        ax_go = plt.axes([0.1, 0.05, 0.12, 0.04])
        self.btn_go = Button(ax_go, 'Mark GO')
        self.btn_go.on_clicked(lambda _: self.mark_go())
        
        ax_initial = plt.axes([0.24, 0.05, 0.12, 0.04])
        self.btn_initial = Button(ax_initial, 'Mark t_initial')
        self.btn_initial.on_clicked(lambda _: self.mark_t_initial())
        
        ax_save = plt.axes([0.38, 0.05, 0.12, 0.04])
        self.btn_save = Button(ax_save, 'Save Labels')
        self.btn_save.on_clicked(lambda _: self.save_labels())
        
        ax_save_next = plt.axes([0.52, 0.05, 0.12, 0.04])
        self.btn_save_next = Button(ax_save_next, 'Save & Next')
        self.btn_save_next.on_clicked(lambda _: self.save_and_next())
        
        ax_next = plt.axes([0.66, 0.05, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_next.on_clicked(lambda _: self.next_recording())
    
    def mark_go(self):
        self.go_frame = self.current_frame
        self.update_title()
        print(f"GO marked at frame {self.go_frame} ({self.trigger_times[self.go_frame]/1e6:.3f}s)")
    
    def mark_t_initial(self):
        self.t_initial_frame = self.current_frame
        self.update_title()
        print(f"t_initial marked at frame {self.t_initial_frame} ({self.trigger_times[self.t_initial_frame]/1e6:.3f}s)")
    
    def save_labels(self):
        if self.go_frame is None or self.t_initial_frame is None:
            print("ERROR: must mark both GO and t_initial before saving")
            return False
        
        labels = {
            'go_frame': self.go_frame,
            'go_time_us': int(self.trigger_times[self.go_frame]),
            't_initial_frame': self.t_initial_frame,
            't_initial_time_us': int(self.trigger_times[self.t_initial_frame]),
            'recording_folder': str(self.recording_folder)
        }
        
        # save to file
        save_path = self.recording_folder / 'labels.npy'
        np.save(save_path, labels)
        print(f"\nLabels saved to {save_path}")
        print(f"  GO: frame {labels['go_frame']} → {labels['go_time_us']} µs")
        print(f"  t_initial: frame {labels['t_initial_frame']} → {labels['t_initial_time_us']} µs")
        return True
    
    def next_recording(self):
        next_folder = self.get_next_recording()
        if next_folder is None:
            print("✓ No more recordings to label!")
            return
        
        print(f"\n→ Loading next recording: {next_folder}")
        
        # Update state in-place instead of creating a new instance
        self.recording_folder = Path(next_folder)
        self.go_frame = None
        self.t_initial_frame = None
        self.current_frame = 0
        
        trigger_path = self.recording_folder / 'basler_frame_timestamps.npy'
        self.trigger_times = np.load(trigger_path)
        self.n_frames = len(self.trigger_times)
        
        self.basler_files = sorted(
            [f for f in os.listdir(self.recording_folder)
            if f.startswith('Basler') and f.endswith('.raw')],
            key=lambda x: int(x.split('__')[1].split('.')[0])
        )[:self.n_frames]
        
        self.load_saved_labels_or_metadata()
        
        # Update slider range
        self.slider.valmax = self.n_frames - 1
        self.slider.ax.set_xlim(0, self.n_frames - 1)
        init_frame = self.go_frame if self.go_frame is not None else 0
        self.slider.set_val(init_frame)
        
        # Reset image display so imshow re-initializes
        self.img_display = None
        self.ax.cla()
        
        self.load_frame(init_frame)

    def get_next_recording(self):
        folder_name = self.recording_folder.name
        prefix = folder_name.split('_')[0]
        current_index = int(folder_name.split('_')[1])
        
        gesture_map = {'r': 'rock', 'p': 'paper', 's': 'scissor'}
        gestures = ['rock', 'paper', 'scissor']
        current_gesture = gesture_map[prefix]
        
        base = self.recording_folder.parent.parent
        
        next_index = current_index + 1
        next_folder = base / current_gesture / f"{prefix}_{next_index}"
        
        if next_folder.exists():
            return next_folder
        
        gesture_idx = gestures.index(current_gesture)
        if gesture_idx < len(gestures) - 1:
            next_gesture = gestures[gesture_idx + 1]
            next_prefix = next_gesture[0]
            next_folder = base / next_gesture / f"{next_prefix}_1"
            if next_folder.exists():
                return next_folder
        
        return None
    
    def save_and_next(self):
        if self.save_labels():
            self.next_recording()
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    tool = GestureLabelingTool('/home/lau/Documents/test_2/rock/r_1')
    tool.show()