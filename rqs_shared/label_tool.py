import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')


class RecordingData:
    """Handles loading and accessing recording data."""
    
    def __init__(self, recording_folder: Path):
        self.folder = Path(recording_folder)
        self.trigger_times = self._load_trigger_times()
        self.n_frames = len(self.trigger_times)
        self.basler_files = self._load_basler_files()
        
    def _load_trigger_times(self) -> np.ndarray:
        trigger_path = self.folder / 'basler_frame_timestamps.npy'
        return np.load(trigger_path)
    
    def _load_basler_files(self) -> list[str]:
        """Load and sort Basler .raw files by frame number."""
        files = [
            f for f in os.listdir(self.folder)
            if f.startswith('Basler') and f.endswith('.raw')
        ]
        sorted_files = sorted(files, key=lambda x: int(x.split('__')[1].split('.')[0]))
        
        if not sorted_files:
            raise FileNotFoundError(f"No Basler .raw files found in {self.folder}")
        
        return sorted_files[:self.n_frames]
    
    def load_frame_image(self, frame_idx: int) -> np.ndarray:
        """Load a single frame as grayscale image array."""
        frame_idx = np.clip(frame_idx, 0, self.n_frames - 1)
        frame_path = self.folder / self.basler_files[frame_idx]
        return np.fromfile(frame_path, dtype=np.uint8).reshape(1200, 1920)
    
    def get_frame_time_ms(self, frame_idx: int) -> float:
        """Get timestamp relative to first frame in milliseconds."""
        return (self.trigger_times[frame_idx] - self.trigger_times[0]) / 1000


class LabelState:
    """Manages label persistence (GO and t_initial frames)."""
    
    def __init__(self, recording_folder: Path):
        self.folder = Path(recording_folder)
        self.go_frame = None
        self.t_initial_frame = None
        self._load()
    
    def _load(self):
        """Load labels from file or metadata."""
        labels_path = self.folder / 'labels.npy'
        
        if labels_path.exists():
            try:
                labels = np.load(labels_path, allow_pickle=True).item()
                self.go_frame = labels.get('go_frame')
                self.t_initial_frame = labels.get('t_initial_frame')
                print(f"✓ Loaded saved labels: GO={self.go_frame}, t_initial={self.t_initial_frame}")
                return
            except Exception as e:
                print(f"⚠ Could not load labels: {e}")
        
        self._load_go_from_metadata()
    
    def _load_go_from_metadata(self):
        """Fallback: load GO frame from recording metadata."""
        metadata_path = self.folder / 'recording_metadata.npy'
        
        if not metadata_path.exists():
            return
        
        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
            expected_go_frame = metadata.get('expected_go_frame')
            
            if expected_go_frame is not None:
                self.go_frame = expected_go_frame
                print(f"✓ GO frame auto-loaded from metadata: {self.go_frame}")
        except Exception as e:
            print(f"⚠ Could not load GO from metadata: {e}")
    
    def save(self, trigger_times: np.ndarray) -> bool:
        """Save labels to disk."""
        if self.go_frame is None or self.t_initial_frame is None:
            print("ERROR: must mark both GO and t_initial before saving")
            return False
        
        labels = {
            'go_frame': self.go_frame,
            'go_time_us': int(trigger_times[self.go_frame]),
            't_initial_frame': self.t_initial_frame,
            't_initial_time_us': int(trigger_times[self.t_initial_frame]),
            'recording_folder': str(self.folder)
        }
        
        save_path = self.folder / 'labels.npy'
        np.save(save_path, labels)
        print(f"\n✓ Labels saved to {save_path}")
        print(f"  GO: frame {labels['go_frame']} → {labels['go_time_us']} µs")
        print(f"  t_initial: frame {labels['t_initial_frame']} → {labels['t_initial_time_us']} µs")
        return True


class RecordingNavigator:
    """Handles navigation between recordings."""
    
    GESTURES = ['rock', 'paper', 'scissor']
    GESTURE_MAP = {'r': 'rock', 'p': 'paper', 's': 'scissor'}
    
    @staticmethod
    def get_next(current_folder: Path) -> Path | None:
        """Find next recording in sequence (same gesture → next gesture)."""
        folder_name = current_folder.name
        prefix = folder_name.split('_')[0]
        current_index = int(folder_name.split('_')[1])
        current_gesture = RecordingNavigator.GESTURE_MAP[prefix]
        
        base = current_folder.parent.parent
        
        # Try next index in same gesture
        next_folder = base / current_gesture / f"{prefix}_{current_index + 1}"
        if next_folder.exists():
            return next_folder
        
        # Try first recording of next gesture
        gesture_idx = RecordingNavigator.GESTURES.index(current_gesture)
        if gesture_idx < len(RecordingNavigator.GESTURES) - 1:
            next_gesture = RecordingNavigator.GESTURES[gesture_idx + 1]
            next_prefix = next_gesture[0]
            next_folder = base / next_gesture / f"{next_prefix}_1"
            if next_folder.exists():
                return next_folder
        
        return None


class GestureLabelingTool:
    """UI for labeling gesture recordings."""
    
    def __init__(self, recording_folder: Path):
        self.data = RecordingData(recording_folder)
        self.labels = LabelState(recording_folder)
        
        self.current_frame = 0
        self.t_initial_marker = None
        
        # Setup matplotlib UI
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        self.img_display = None
        self._setup_controls()
        
        # Jump to GO frame or start at beginning
        init_frame = self.labels.go_frame if self.labels.go_frame is not None else 0
        self._display_frame(init_frame)
    
    def _setup_controls(self):
        """Create slider and buttons."""
        # Frame slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(
            ax_slider, 'Frame', 0, self.data.n_frames - 1,
            valinit=self.labels.go_frame if self.labels.go_frame is not None else 0,
            valstep=1
        )
        self.slider.on_changed(self._on_slider_change)
        
        # Buttons
        buttons = [
            (0.1, 'Mark GO', self._mark_go),
            (0.24, 'Mark t_initial', self._mark_t_initial),
            (0.38, 'Save Labels', self._save_labels),
            (0.52, 'Save & Next', self._save_and_next),
            (0.66, 'Next →', self._next_recording),
        ]
        
        for x_pos, label, callback in buttons:
            ax_btn = plt.axes([x_pos, 0.05, 0.12, 0.04])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda _, cb=callback: cb())
    
    def _display_frame(self, frame_idx: int):
        """Load and display a frame."""
        frame_idx = np.clip(frame_idx, 0, self.data.n_frames - 1)
        
        try:
            frame = self.data.load_frame_image(frame_idx)
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return
        
        # Update image display
        if self.img_display is None:
            self.img_display = self.ax.imshow(frame, cmap='gray')
        else:
            self.img_display.set_data(frame)
        
        self.current_frame = frame_idx
        self._update_title()
        self._update_t_initial_marker()
        
        # Sync slider without triggering callback
        if int(self.slider.val) != frame_idx:
            self.slider.set_val(frame_idx)
        
        self.fig.canvas.draw_idle()
    
    def _update_title(self):
        """Update plot title with current state."""
        time_ms = self.data.get_frame_time_ms(self.current_frame)
        
        parts = [
            f"{self.data.folder.name}",
            f"Frame {self.current_frame}/{self.data.n_frames-1}",
            f"Time: {time_ms:.1f}ms"
        ]
        
        if self.labels.go_frame is not None:
            parts.append(f"GO: frame {self.labels.go_frame}")
        if self.labels.t_initial_frame is not None:
            parts.append(f"t_initial: frame {self.labels.t_initial_frame}")
        
        title = " | ".join(parts)
        title += "\nKeys: ←/→ frame, ↓ GO, ↑ t_initial, Shift save+next"
        
        self.ax.set_title(title)
    
    def _update_t_initial_marker(self):
        """Draw/update red line at t_initial position on slider."""
        if self.t_initial_marker is not None:
            self.t_initial_marker.remove()
            self.t_initial_marker = None
        
        if self.labels.t_initial_frame is not None:
            self.t_initial_marker = self.slider.ax.axvline(
                self.labels.t_initial_frame,
                color='red',
                linestyle='--',
                linewidth=2
            )
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key_actions = {
            'left': lambda: self._display_frame(self.current_frame - 1),
            'right': lambda: self._display_frame(self.current_frame + 1),
            'down': self._mark_go,
            'up': self._mark_t_initial,
            'shift': self._save_and_next,
        }
        
        action = key_actions.get(event.key)
        if action:
            action()
    
    def _on_slider_change(self, val):
        """Handle slider movement."""
        self._display_frame(int(val))
    
    def _mark_go(self):
        """Mark current frame as GO."""
        self.labels.go_frame = self.current_frame
        self._update_title()
        print(f"GO marked at frame {self.labels.go_frame}")
    
    def _mark_t_initial(self):
        """Mark current frame as t_initial."""
        self.labels.t_initial_frame = self.current_frame
        self._update_title()
        self._update_t_initial_marker()
        print(f"t_initial marked at frame {self.labels.t_initial_frame}")
    
    def _save_labels(self):
        """Save current labels to disk."""
        return self.labels.save(self.data.trigger_times)
    
    def _next_recording(self):
        """Load next recording in sequence."""
        next_folder = RecordingNavigator.get_next(self.data.folder)
        
        if next_folder is None:
            print("✓ No more recordings to label!")
            return
        
        print(f"\n→ Loading next recording: {next_folder}")
        
        # Reload data and labels
        self.data = RecordingData(next_folder)
        self.labels = LabelState(next_folder)
        self.current_frame = 0
        
        # Update slider range
        self.slider.valmax = self.data.n_frames - 1
        self.slider.ax.set_xlim(0, self.data.n_frames - 1)
        
        # Reset display
        self.img_display = None
        self.ax.cla()
        
        init_frame = self.labels.go_frame if self.labels.go_frame is not None else 0
        self._display_frame(init_frame)
    
    def _save_and_next(self):
        """Save labels and advance to next recording."""
        if self._save_labels():
            self._next_recording()
    
    def show(self):
        """Display the tool window."""
        plt.show()


if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    tool = GestureLabelingTool(base / "rock" / "r_1")
    tool.show()