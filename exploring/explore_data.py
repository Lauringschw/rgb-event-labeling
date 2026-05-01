import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

GESTURES   = ['rock', 'paper', 'scissor']
COLORS     = {'rock': '#e74c3c', 'paper': '#3498db', 'scissor': '#2ecc71'}
N_SAMPLES  = 5   # recordings per gesture to average over


def load_labels(folder: Path):
    p = folder / 'labels.npy'
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True).item()


def load_events(folder: Path):
    raw = folder / 'prophesee_events.raw'
    if not raw.exists():
        return None
    ev_list = []
    for evs in EventsIterator(str(raw)):
        ev_list.append(evs)
    return np.concatenate(ev_list)


def to_histogram(events, height=720, width=1280):
    """2D ON-event count histogram, 3-sigma normalised to [0, 1]."""
    frame = np.zeros((height, width), dtype=np.float32)
    on = events[events['p'] == 1]
    np.add.at(frame, (on['y'], on['x']), 1)
    frame = np.clip(frame, 0, 200)
    std = frame.std()
    if std > 0:
        frame = (frame - frame.mean()) / (3 * std + 1e-8)
        frame = np.clip(frame, 0, 1)
    return frame


def get_recordings(gesture_dir: Path, n: int):
    folders = sorted([f for f in gesture_dir.iterdir() if f.is_dir()])
    return folders[:n]


def figure_event_density(base: Path, output_dir: Path):
    BIN_MS    = 10
    WINDOW_MS = 500
    bin_us    = BIN_MS * 1000
    n_bins    = WINDOW_MS // BIN_MS

    fig, ax = plt.subplots(figsize=(10, 5))

    for gesture in GESTURES:
        folders = get_recordings(base / gesture, N_SAMPLES)
        all_counts   = []
        t_initials   = []

        for folder in folders:
            labels = load_labels(folder)
            if labels is None:
                continue
            events = load_events(folder)
            if events is None:
                continue

            t_go      = labels['go_time_us']
            t_initial = labels['t_initial_time_us']
            t_initials.append((t_initial - t_go) / 1000)

            mask = (events['t'] >= t_go) & (events['t'] < t_go + WINDOW_MS * 1000)
            ev   = events[mask]

            counts = np.zeros(n_bins, dtype=np.float32)
            for i in range(n_bins):
                lo = t_go + i * bin_us
                hi = lo + bin_us
                counts[i] = np.sum((ev['t'] >= lo) & (ev['t'] < hi))
            all_counts.append(counts)

        if not all_counts:
            continue

        centers = np.arange(n_bins) * BIN_MS + BIN_MS / 2
        mean_counts = np.mean(all_counts, axis=0)
        std_counts  = np.std(all_counts, axis=0)

        color = COLORS[gesture]
        ax.plot(centers, mean_counts, color=color, linewidth=2, label=gesture.capitalize())
        ax.fill_between(centers,
                        mean_counts - std_counts,
                        mean_counts + std_counts,
                        color=color, alpha=0.15)

        ti_mean = np.mean(t_initials)
        ti_std  = np.std(t_initials)
        ax.axvspan(ti_mean - ti_std, ti_mean + ti_std,
                   color=color, alpha=0.08)
        ax.axvline(ti_mean, color=color, linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Time from GO signal (ms)', fontsize=12)
    ax.set_ylabel(f'Event count per {BIN_MS}ms bin', fontsize=12)
    ax.set_title('Event Density Over Time by Gesture\n'
                 '(mean ± std, dashed line = mean t_initial)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / 'event_density_over_time.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


def figure_gesture_window_grid(base: Path, output_dir: Path):
    WINDOWS_MS = [50, 100, 150, 200]

    fig, axes = plt.subplots(len(GESTURES), len(WINDOWS_MS),
                             figsize=(4 * len(WINDOWS_MS), 4 * len(GESTURES)))

    for row, gesture in enumerate(GESTURES):
        folders = get_recordings(base / gesture, N_SAMPLES)

        for col, window_ms in enumerate(WINDOWS_MS):
            ax    = axes[row, col]
            stack = []

            for folder in folders:
                labels = load_labels(folder)
                if labels is None:
                    continue
                events = load_events(folder)
                if events is None:
                    continue

                t0   = labels['t_initial_time_us']
                mask = (events['t'] >= t0) & (events['t'] < t0 + window_ms * 1000)
                ev   = events[mask]
                if len(ev) == 0:
                    continue

                stack.append(to_histogram(ev))

            if stack:
                avg_frame   = np.mean(stack, axis=0)
                event_count = int(np.mean([
                    np.sum((load_events(f)['t'] >= load_labels(f)['t_initial_time_us']) &
                           (load_events(f)['t'] <  load_labels(f)['t_initial_time_us'] + window_ms * 1000))
                    for f in folders
                    if load_labels(f) is not None and load_events(f) is not None
                ]))
            else:
                avg_frame   = np.zeros((720, 1280), dtype=np.float32)
                event_count = 0

            im = ax.imshow(avg_frame, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax.axis('off')

            ax.text(0.02, 0.02, f'~{event_count:,} events',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                    color='white')

            if row == 0:
                ax.set_title(f'{window_ms} ms', fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(gesture.capitalize(), fontsize=13,
                              fontweight='bold', rotation=0, labelpad=50)
            if col == len(WINDOWS_MS) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Event Histograms: Gesture × Window Length (anchored at t_initial)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = output_dir / 'gesture_window_grid.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


def figure_offset_window_heatmap(base: Path, output_dir: Path):
    OFFSETS_MS = [0, 25, 50, 75, 100]
    WINDOWS_MS = [20, 50, 100, 150, 200]

    fig, axes = plt.subplots(1, len(GESTURES),
                             figsize=(5 * len(GESTURES), 5))

    for idx, gesture in enumerate(GESTURES):
        ax      = axes[idx]
        folders = get_recordings(base / gesture, N_SAMPLES)
        matrix  = np.zeros((len(OFFSETS_MS), len(WINDOWS_MS)), dtype=np.float32)

        for folder in folders:
            labels = load_labels(folder)
            if labels is None:
                continue
            events = load_events(folder)
            if events is None:
                continue
            t0 = labels['t_initial_time_us']

            for i, offset_ms in enumerate(OFFSETS_MS):
                for j, window_ms in enumerate(WINDOWS_MS):
                    lo = t0 + offset_ms * 1000
                    hi = lo + window_ms * 1000
                    matrix[i, j] += np.sum((events['t'] >= lo) & (events['t'] < hi))

        matrix /= max(len(folders), 1)

        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(WINDOWS_MS)))
        ax.set_xticklabels([f'{w}ms' for w in WINDOWS_MS], fontsize=9)
        ax.set_yticks(range(len(OFFSETS_MS)))
        ax.set_yticklabels([f'+{o}ms' for o in OFFSETS_MS], fontsize=9)
        ax.set_xlabel('Window length', fontsize=11)
        ax.set_ylabel('Offset from t_initial', fontsize=11)
        ax.set_title(gesture.capitalize(), fontsize=13, fontweight='bold')

        for i in range(len(OFFSETS_MS)):
            for j in range(len(WINDOWS_MS)):
                ax.text(j, i, f'{matrix[i, j]:,.0f}',
                        ha='center', va='center',
                        fontsize=7, color='white')

        plt.colorbar(im, ax=ax, label='Avg event count')

    fig.suptitle('Average Event Count: Offset × Window Length',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = output_dir / 'offset_window_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


if __name__ == '__main__':
    base       = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    output_dir = Path(os.getenv("EXPLORATION_DIR"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('DATA EXPLORATION')
    print('=' * 60)
    print(f'Input:   {base}')
    print(f'Output:  {output_dir}')
    print(f'Samples: {N_SAMPLES} recordings per gesture')
    print()

    print('Figure 1: event density over time...')
    figure_event_density(base, output_dir)

    print('Figure 2: gesture × window grid...')
    figure_gesture_window_grid(base, output_dir)

    print('Figure 3: offset × window heatmap...')
    figure_offset_window_heatmap(base, output_dir)

    print()
    print('=' * 60)
    print(f'Done. Figures saved to {output_dir}/')
    print('=' * 60)
