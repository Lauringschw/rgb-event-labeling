import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

class GestureLabelingTool:
    def __init__(self, recording_folder):
        self.recording_folder = recording_folder
        
        # load trigger timestamps
        self.trigger_times = np.load('basler_frame_timestamps.npy')
        self.n_frames = len(self.trigger_times)
        
        # load basler files
        self.basler_files = sorted([
            f for f in os.listdir(recording_folder) 
            if f.startswith('Basler') and f.endswith('.raw')
        ])[:self.n_frames]  # only use frames with triggers
        
        self.current_frame = 0
        self.go_frame = None
        self.t_initial_frame = None
        
        # setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        self.img_display = None
        self.setup_ui()
        self.load_frame(0)
        
    def load_frame(self, frame_idx):
        frame_path = os.path.join(self.recording_folder, self.basler_files[frame_idx])
        frame = np.fromfile(frame_path, dtype=np.uint8).reshape(1200, 1920)
        
        if self.img_display is None:
            self.img_display = self.ax.imshow(frame, cmap='gray')
        else:
            self.img_display.set_data(frame)
        
        self.current_frame = frame_idx
        self.update_title()
        self.fig.canvas.draw_idle()
    
    def update_title(self):
        title = f"Frame {self.current_frame}/{self.n_frames-1} | "
        title += f"Time: {self.trigger_times[self.current_frame]/1e6:.3f}s"
        
        if self.go_frame is not None:
            title += f" | GO: frame {self.go_frame}"
        if self.t_initial_frame is not None:
            title += f" | t_initial: frame {self.t_initial_frame}"
        
        self.ax.set_title(title)
    
    def setup_ui(self):
        # slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames-1, 
                             valinit=0, valstep=1)
        self.slider.on_changed(lambda val: self.load_frame(int(val)))
        
        # buttons
        ax_go = plt.axes([0.15, 0.05, 0.15, 0.04])
        self.btn_go = Button(ax_go, 'Mark GO')
        self.btn_go.on_clicked(lambda _: self.mark_go())
        
        ax_initial = plt.axes([0.35, 0.05, 0.15, 0.04])
        self.btn_initial = Button(ax_initial, 'Mark t_initial')
        self.btn_initial.on_clicked(lambda _: self.mark_t_initial())
        
        ax_save = plt.axes([0.7, 0.05, 0.15, 0.04])
        self.btn_save = Button(ax_save, 'Save Labels')
        self.btn_save.on_clicked(lambda _: self.save_labels())
    
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
            return
        
        labels = {
            'go_frame': self.go_frame,
            'go_time_us': int(self.trigger_times[self.go_frame]),
            't_initial_frame': self.t_initial_frame,
            't_initial_time_us': int(self.trigger_times[self.t_initial_frame]),
            'recording_folder': self.recording_folder
        }
        
        # save to file
        save_path = os.path.join(self.recording_folder, 'labels.npy')
        np.save(save_path, labels)
        print(f"\nLabels saved to {save_path}")
        print(f"  GO: frame {labels['go_frame']} → {labels['go_time_us']} µs")
        print(f"  t_initial: frame {labels['t_initial_frame']} → {labels['t_initial_time_us']} µs")
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    tool = GestureLabelingTool('/home/lau/Documents/test_1/rock/r_3')
    tool.show()