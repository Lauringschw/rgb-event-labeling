import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# Load environment
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    output_dir = Path(os.getenv("EXPLORATION_DIR")) / "go_to_tinitial"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Analyzing recordings in: {base}")
print(f"Output directory: {output_dir}")

if not base.exists():
    print(f"Error: directory not found")
    exit(1)

delays = []
delays_by_gesture = {'rock': [], 'paper': [], 'scissor': []}

for gesture in ['rock', 'paper', 'scissor']:
    gesture_dir = base / gesture
    if not gesture_dir.exists():
        continue
    
    for recording in gesture_dir.iterdir():
        if not recording.is_dir():
            continue
        
        metadata_path = recording / "recording_metadata.npy"
        labels_path = recording / "labels.npy"
        basler_timestamps_path = recording / "basler_frame_timestamps.npy"
        
        if not all([metadata_path.exists(), labels_path.exists(), basler_timestamps_path.exists()]):
            continue
        
        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
            labels = np.load(labels_path, allow_pickle=True).item()
            basler_timestamps = np.load(basler_timestamps_path)
            
            go_offset_seconds = metadata.get('go_offset_from_start')
            if go_offset_seconds is None:
                go_timestamp_system = metadata.get('go_timestamp_system')
                recording_start_time = metadata.get('recording_start_time')
                if go_timestamp_system and recording_start_time:
                    go_offset_seconds = go_timestamp_system - recording_start_time
            
            t_initial_prophesee = labels.get('t_initial_time_us')
            recording_start_prophesee_us = basler_timestamps[0]
            
            if go_offset_seconds and t_initial_prophesee:
                go_offset_us = go_offset_seconds * 1_000_000
                go_time_prophesee_us = recording_start_prophesee_us + go_offset_us
                
                delay_ms = (t_initial_prophesee - go_time_prophesee_us) / 1000.0
                delays.append(delay_ms)
                delays_by_gesture[gesture].append(delay_ms)
        except:
            continue

delays = np.array(delays)

# Statistics
stats = {
    'n': len(delays),
    'mean': delays.mean(),
    'median': np.median(delays),
    'std': delays.std(),
    'min': delays.min(),
    'max': delays.max(),
    'p25': np.percentile(delays, 25),
    'p50': np.percentile(delays, 50),
    'p75': np.percentile(delays, 75),
    'p90': np.percentile(delays, 90)
}

# Per-gesture stats
gesture_stats = {}
for gesture in ['rock', 'paper', 'scissor']:
    g_delays = np.array(delays_by_gesture[gesture])
    gesture_stats[gesture] = {
        'n': len(g_delays),
        'mean': g_delays.mean(),
        'std': g_delays.std()
    }

# Print summary
print(f"\nAnalyzed {stats['n']} recordings")
print(f"Mean: {stats['mean']:.1f} ms")
print(f"Median: {stats['median']:.1f} ms")
print(f"Std dev: {stats['std']:.1f} ms")
print(f"Range: [{stats['min']:.1f}, {stats['max']:.1f}] ms")
print(f"\nPer-gesture:")
for gesture in ['rock', 'paper', 'scissor']:
    gs = gesture_stats[gesture]
    print(f"  {gesture:8s}: {gs['mean']:6.1f} ± {gs['std']:5.1f} ms (n={gs['n']})")

# Save statistics
np.save(output_dir / 'go_to_tinitial_stats.npy', {
    'overall': stats,
    'by_gesture': gesture_stats,
    'raw_delays': delays,
    'delays_by_gesture': delays_by_gesture
})

# ===== VISUALIZATIONS =====

# 1. Distribution histogram with per-gesture overlays
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(-600, 700, 50)

for gesture, color in [('rock', '#E74C3C'), ('paper', '#3498DB'), ('scissor', '#2ECC71')]:
    g_delays = np.array(delays_by_gesture[gesture])
    ax.hist(g_delays, bins=bins, alpha=0.5, label=gesture.capitalize(), color=color, edgecolor='black')

ax.axvline(0, color='black', linestyle='--', linewidth=2, label='GO signal', alpha=0.7)
ax.axvline(stats['median'], color='red', linestyle='-', linewidth=2, label=f'Median ({stats["median"]:.0f}ms)')

ax.set_xlabel('Time from GO to t_initial (ms)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('GO → t_initial Delay Distribution by Gesture', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.text(0.02, 0.98, f'n={stats["n"]} recordings\nNegative = motion before GO', 
        transform=ax.transAxes, verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'go_to_tinitial_distribution.png', dpi=150)
plt.close()

# 2. Box plot comparison
fig, ax = plt.subplots(figsize=(8, 6))

data_for_box = [delays_by_gesture['rock'], delays_by_gesture['paper'], delays_by_gesture['scissor']]
bp = ax.boxplot(data_for_box, labels=['Rock', 'Paper', 'Scissor'], patch_artist=True,
                widths=0.6, showmeans=True, meanline=True)

colors = ['#E74C3C', '#3498DB', '#2ECC71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='GO signal', alpha=0.7)
ax.set_ylabel('Time from GO to t_initial (ms)', fontsize=12)
ax.set_title('Delay Distribution by Gesture Type', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'go_to_tinitial_boxplot.png', dpi=150)
plt.close()

# 3. CDF (cumulative distribution)
fig, ax = plt.subplots(figsize=(10, 6))

sorted_delays = np.sort(delays)
cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays) * 100

ax.plot(sorted_delays, cdf, linewidth=2, color='#34495E')
ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='GO signal', alpha=0.7)
ax.axhline(50, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axvline(stats['median'], color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax.plot([stats['median']], [50], 'ro', markersize=10, label=f'Median: {stats["median"]:.0f}ms')

ax.set_xlabel('Time from GO to t_initial (ms)', fontsize=12)
ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax.set_title('Cumulative Distribution: When Do Gestures Start?', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# Annotations
ax.annotate(f'{stats["p75"]:.0f}ms\n(75th percentile)', 
            xy=(stats['p75'], 75), xytext=(stats['p75']+150, 85),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'go_to_tinitial_cdf.png', dpi=150)
plt.close()