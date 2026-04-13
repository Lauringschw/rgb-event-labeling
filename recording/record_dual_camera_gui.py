from pypylon import pylon
from metavision_hal import DeviceDiscovery, I_TriggerIn
import numpy as np
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

class RecordingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Camera Recording")
        self.root.geometry("600x750")
        
        self.recording_num = 1
        self.output_dir = None
        self.camera_basler = None
        self.device = None
        self.i_events_stream = None
        self.stop_recording = False
        self.go_timestamp_system = None
        self.recording_start_time = None
        self.frame_idx = 0
        self.basler_timestamps = []
        
        self.setup_ui()
        self.initialize_cameras()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Dual Camera Recording", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Layered display frame for preview + countdown overlay
        display_frame = tk.Frame(main_frame, bg='black')
        display_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.preview_label = tk.Label(display_frame, bg='black')
        self.preview_label.pack()

        self.countdown_label = tk.Label(display_frame, text="", 
                                       font=('Arial', 120, 'bold'),
                                       fg='red', bg='black')
        self.countdown_label.place(relx=0.5, rely=0.5, anchor='center')
        self.countdown_label.lower()

        # Recording settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Recording Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Gesture selection
        ttk.Label(settings_frame, text="Gesture:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.gesture_var = tk.StringVar(value="rock")
        gesture_combo = ttk.Combobox(settings_frame, textvariable=self.gesture_var, 
                                     values=["rock", "paper", "scissor", "other"], 
                                     state="readonly", width=15)
        gesture_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Recording number
        ttk.Label(settings_frame, text="Recording #:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.recording_num_var = tk.StringVar(value="1")
        ttk.Entry(settings_frame, textvariable=self.recording_num_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Base directory
        ttk.Label(settings_frame, text="Base folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.base_dir_var = tk.StringVar(value=Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR")))
        ttk.Entry(settings_frame, textvariable=self.base_dir_var, width=30).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Current output path display
        ttk.Label(settings_frame, text="Will save to:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_path_label = ttk.Label(settings_frame, text="", foreground="blue")
        self.output_path_label.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Update output path when gesture or number changes
        self.gesture_var.trace('w', lambda *args: self.update_output_path())
        self.recording_num_var.trace('w', lambda *args: self.update_output_path())
        self.update_output_path()
        
        # Status display
        self.status_text = tk.Text(main_frame, height=12, width=60, state='disabled', 
                                   bg='#f0f0f0', font=('Courier', 9))
        self.status_text.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=520, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Step buttons
        self.btn_single = ttk.Button(btn_frame, text="1. Single Shot", 
                                     command=self.capture_single, width=20)
        self.btn_single.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_start = ttk.Button(btn_frame, text="2. Start Recording", 
                                    command=self.start_recording, width=20, state='disabled')
        self.btn_start.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="Stop Recording", 
                                   command=self.stop_recording_manual, width=20, state='disabled')
        self.btn_stop.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Bias settings
        bias_frame = ttk.LabelFrame(main_frame, text="Prophesee Bias Settings", padding="10")
        bias_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(bias_frame, text="bias_diff_on:").grid(row=0, column=0, sticky=tk.W)
        self.bias_on_var = tk.StringVar(value="32")
        ttk.Entry(bias_frame, textvariable=self.bias_on_var, width=10).grid(row=0, column=1)
        
        ttk.Label(bias_frame, text="bias_diff_off:").grid(row=1, column=0, sticky=tk.W)
        self.bias_off_var = tk.StringVar(value="32")
        ttk.Entry(bias_frame, textvariable=self.bias_off_var, width=10).grid(row=1, column=1)
    
    def update_output_path(self):
        """Update the output path display"""
        try:
            gesture = self.gesture_var.get()
            num = int(self.recording_num_var.get())
            prefix = gesture[0]  # r, p, or s
            path = f"{gesture}/{prefix}_{num}"
            self.output_path_label.config(text=path)
        except:
            self.output_path_label.config(text="invalid number")
        
    def show_countdown(self, text, color='red'):
        """Show countdown overlay"""
        self.countdown_label.config(text=text, fg=color)
        self.countdown_label.lift()
        self.root.update()
        
    def hide_countdown(self):
        """Hide countdown overlay"""
        self.countdown_label.config(text="")
        self.countdown_label.lower()
        self.root.update()
        
    def log(self, message):
        """Add message to status text"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message + '\n')
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
        self.root.update()
        
    def initialize_cameras(self):
        """Initialize both cameras"""
        try:
            self.log("Initializing Basler camera...")
            self.camera_basler = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera_basler.Open()
            
            # Load UserSet1
            try:
                self.camera_basler.UserSetSelector.SetValue("UserSet1")
                self.camera_basler.UserSetLoad.Execute()
                self.log("✓ Loaded UserSet1")
            except:
                self.log("⚠ Could not load UserSet1")
            
            # Configure for continuous
            self.camera_basler.AcquisitionMode.SetValue("Continuous")
            self.camera_basler.AcquisitionFrameRateEnable.SetValue(True)
            self.camera_basler.AcquisitionFrameRate.SetValue(140.0)
            
            # Configure trigger
            try:
                self.camera_basler.LineSelector.SetValue("Line2")
                self.camera_basler.LineMode.SetValue("Output")
                self.camera_basler.LineSource.SetValue("ExposureActive")
                self.camera_basler.LineInverter.SetValue(False)
                self.log("✓ Trigger output configured")
            except Exception as e:
                self.log(f"⚠ Trigger setup failed: {e}")
            
            self.log("✓ Basler initialized")
            self.log("\nReady for single shot capture")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize cameras:\n{e}")
            
    def capture_single(self):
        """Capture single calibration frame"""
        try:
            # Build output path: base_dir/gesture/prefix_number/
            gesture = self.gesture_var.get()
            recording_num = int(self.recording_num_var.get())
            prefix = gesture[0]  # 'r', 'p', or 's'
            
            base_dir = Path(self.base_dir_var.get())
            self.output_dir = base_dir / gesture / f"{prefix}_{recording_num}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.log(f"\n📸 Capturing single frame to {self.output_dir}...")
            self.btn_single.config(state='disabled')
            
            self.camera_basler.AcquisitionMode.SetValue("SingleFrame")
            self.camera_basler.StartGrabbing(pylon.GrabStrategy_OneByOne)
            grab_result = self.camera_basler.RetrieveResult(5000)
            
            if grab_result.GrabSucceeded():
                calib_img = grab_result.Array
                calib_path = self.output_dir / "calibration_frame.raw"
                calib_img.tofile(calib_path)
                self.log(f"✓ Calibration frame saved")
            
            grab_result.Release()
            self.camera_basler.StopGrabbing()
            
            # Re-configure for continuous
            self.camera_basler.AcquisitionMode.SetValue("Continuous")
            
            self.log("\n✓ Ready to start recording")
            self.btn_start.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Single shot failed:\n{e}")
            self.btn_single.config(state='normal')
            
    def start_recording(self):
        """Start dual camera recording with countdown"""
        try:
            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='normal')
            
            # Initialize Prophesee
            self.log("\n🔴 Initializing Prophesee...")
            self.device = DeviceDiscovery.open("")
            
            if self.device is None:
                raise Exception("Could not open Prophesee camera!")
            
            # Configure biases
            try:
                i_ll_biases = self.device.get_i_ll_biases()
                if i_ll_biases:
                    bias_on = int(self.bias_on_var.get())
                    bias_off = int(self.bias_off_var.get())
                    i_ll_biases.set("bias_diff_on", bias_on)
                    i_ll_biases.set("bias_diff_off", bias_off)
                    self.log(f"✓ Biases set: {bias_on}, {bias_off}")
            except Exception as e:
                self.log(f"⚠ Bias config: {e}")
            
            # Configure trigger
            try:
                i_trigger_in = self.device.get_i_trigger_in()
                if i_trigger_in:
                    i_trigger_in.enable(I_TriggerIn.Channel.MAIN)
                    self.log("✓ Trigger enabled")
            except Exception as e:
                self.log(f"⚠ Trigger: {e}")
            
            # Start recording
            prophesee_output = str(self.output_dir / "prophesee_events.raw")
            self.i_events_stream = self.device.get_i_events_stream()
            self.i_events_stream.log_raw_data(prophesee_output)
            self.i_events_stream.start()
            self.log("✓ Prophesee recording started")
            
            # Start Basler
            self.camera_basler.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.log("✓ Basler recording started")
            
            # Reset counters
            self.frame_idx = 0
            self.basler_timestamps = []
            self.stop_recording = False
            self.recording_start_time = time.time()
            
            # Start background threads
            self.start_background_threads()
            
            # Start countdown sequence
            self.progress.start()
            threading.Thread(target=self.countdown_sequence, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording:\n{e}")
            self.btn_start.config(state='normal')
            self.btn_stop.config(state='disabled')
            
    def start_background_threads(self):
        """Start Prophesee polling and Basler frame grabbing"""
        def prophesee_poll():
            while not self.stop_recording:
                try:
                    self.i_events_stream.get_latest_raw_data()
                    time.sleep(0.001)
                except:
                    break
        
        def grab_frames():
            while not self.stop_recording:
                try:
                    if self.camera_basler.IsGrabbing():
                        grab_result = self.camera_basler.RetrieveResult(100, pylon.TimeoutHandling_Return)
                        
                        if grab_result and grab_result.GrabSucceeded():
                            img = grab_result.Array
                            frame_path = self.output_dir / f"Basler_acA1920-155um__{self.frame_idx}.raw"
                            img.tofile(frame_path)
                            
                            if self.frame_idx % 5 == 0:
                                small = Image.fromarray(img).resize((640, 400))
                                photo = ImageTk.PhotoImage(small)
                                self.preview_label.config(image=photo)
                                self.preview_label.image = photo  # Keep reference
                            
                            timestamp_us = int(grab_result.TimeStamp)
                            self.basler_timestamps.append(timestamp_us)
                            
                            self.frame_idx += 1
                            grab_result.Release()
                except:
                    break
        
        threading.Thread(target=prophesee_poll, daemon=True).start()
        threading.Thread(target=grab_frames, daemon=True).start()
        
    def countdown_sequence(self):
        """Countdown: 2s wait, 3-2-1, GO, 2s wait"""
        try:
            self.log("\n⏱ Recording 1 seconds...")
            time.sleep(1.0)
            
            # Countdown with large display
            self.show_countdown("3", 'yellow')
            time.sleep(1.0)
            
            self.show_countdown("2", 'orange')
            time.sleep(1.0)
            
            self.show_countdown("1", 'red')
            time.sleep(1.0)
            
            self.show_countdown("GO!", 'lime')
            self.log("GO! 🎯")
            
            self.go_timestamp_system = time.time()
            
            time.sleep(1.0)
            
            self.hide_countdown()
            self.log("\n✓ Recording complete")
            
            # Auto-stop
            self.root.after(100, self.finish_recording)
            
        except Exception as e:
            self.log(f"⚠ Error: {e}")
            self.hide_countdown()
            
    def stop_recording_manual(self):
        """Manual stop button"""
        self.hide_countdown()
        self.finish_recording()
        
    def finish_recording(self):
        """Stop cameras and save data"""
        try:
            self.progress.stop()
            self.log("\n⏹ Stopping cameras...")
            
            self.stop_recording = True
            time.sleep(0.5)
            
            # Clear preview image
            self.preview_label.config(image='')
            self.preview_label.image = None
            
            self.camera_basler.StopGrabbing()
            end_time = time.time()
            self.log(f"✓ Basler stopped ({self.frame_idx} frames)")
            
            self.log("  Flushing Prophesee buffer...")
            time.sleep(2)
            
            # Stop and close Prophesee properly
            if self.i_events_stream is not None:
                try:
                    self.i_events_stream.stop()
                    self.i_events_stream.stop_log_raw_data()
                except:
                    pass
                self.i_events_stream = None
            
            # Explicitly delete device reference
            if self.device is not None:
                del self.device
                self.device = None
                time.sleep(0.5)  # Give it time to release
            
            self.log("✓ Prophesee stopped")
            
            # Save data
            timestamps_path = self.output_dir / "basler_frame_timestamps.npy"
            np.save(timestamps_path, np.array(self.basler_timestamps))
            
            if self.go_timestamp_system:
                metadata = {
                    'go_timestamp_system': self.go_timestamp_system,
                    'recording_start_time': self.recording_start_time,
                    'go_offset_from_start': self.go_timestamp_system - self.recording_start_time,
                    'recording_end_time': end_time,
                    'total_frames': self.frame_idx,
                    'expected_go_frame': int((self.go_timestamp_system - self.recording_start_time) * 140)
                }
                metadata_path = self.output_dir / "recording_metadata.npy"
                np.save(metadata_path, metadata)
                
                elapsed = end_time - self.recording_start_time
                self.log(f"\n✓ Recording complete!")
                self.log(f"  Duration: {elapsed:.1f}s")
                self.log(f"  GO at: {metadata['go_offset_from_start']:.3f}s")
                self.log(f"  GO frame: ~{metadata['expected_go_frame']}")
                self.log(f"  Frames: {self.frame_idx}")
                self.log(f"  Saved to: {self.output_dir}")
            
            # Reset UI
            self.btn_single.config(state='normal')
            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='disabled')
            
            # Increment recording number
            self.recording_num_var.set(str(int(self.recording_num_var.get()) + 1))
            self.update_output_path()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error finishing recording:\n{e}")
            
    def on_closing(self):
        """Cleanup on window close"""
        if self.camera_basler:
            try:
                self.camera_basler.Close()
            except:
                pass
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = RecordingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()