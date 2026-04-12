import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv(Path(__file__).parent.parent / ".env")


GESTURES = ["rock", "paper", "scissor"]
RQ_TO_FILE = {
	"rq1": "event_samples_rq1.npy",
	"rq2": "event_samples_rq2.npy",
	"rq3": "event_samples_rq3.npy",
}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Preview extracted event samples and save PNGs to SAMPLES_DIR."
	)
	parser.add_argument(
		"--rq",
		choices=["rq1", "rq2", "rq3", "all"],
		default="all",
		help="Which research-question sample set to preview.",
	)
	parser.add_argument(
		"--num",
		type=int,
		default=6,
		help="How many recordings to preview per RQ.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible sample picking.",
	)
	parser.add_argument(
		"--save-only",
		action="store_true",
		help="Save PNGs without opening preview windows.",
	)
	return parser.parse_args()


def get_base_dir():
	recordings_dir = os.getenv("RECORDINGS_DIR")
	sub_dir = os.getenv("DIR")

	if not recordings_dir or not sub_dir:
		raise ValueError("Please set RECORDINGS_DIR and DIR in .env")

	base_dir = Path(recordings_dir) / Path(sub_dir)
	if not base_dir.exists():
		raise FileNotFoundError(f"Recording base folder does not exist: {base_dir}")

	return base_dir


def get_preview_dir(rq):
	samples_dir = os.getenv("SAMPLES_DIR")
	root = Path(samples_dir) if samples_dir else Path.cwd() / "samples"
	out_dir = root / "previews" / rq
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


def collect_sample_files(base_dir, sample_filename):
	files = []
	for gesture in GESTURES:
		gesture_dir = base_dir / gesture
		if not gesture_dir.exists():
			continue

		for recording_dir in sorted([d for d in gesture_dir.iterdir() if d.is_dir()]):
			sample_file = recording_dir / sample_filename
			if sample_file.exists():
				files.append((gesture, recording_dir.name, sample_file))
	return files


def draw_array(ax, arr, title):
	# Signed histograms are easier to read with a centered diverging colormap.
	vmax = np.max(np.abs(arr))
	if vmax == 0:
		vmax = 1.0
	im = ax.imshow(arr, cmap="seismic", vmin=-vmax, vmax=vmax)
	ax.set_title(title)
	ax.axis("off")
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_sample_dict(gesture, recording_name, sample_dict, rq, out_dir):
	keys = list(sample_dict.keys())
	cols = len(keys)
	fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5), squeeze=False)

	for idx, key in enumerate(keys):
		ax = axes[0, idx]
		value = sample_dict[key]

		if isinstance(value, np.ndarray) and value.ndim == 3:
			# For voxel grids, visualize a temporal sum projection for quick inspection.
			projection = value.sum(axis=0)
			draw_array(ax, projection, f"{key} (sum over bins)")
		else:
			draw_array(ax, value, str(key))

	fig.suptitle(f"{rq.upper()} | {gesture}/{recording_name}")
	fig.tight_layout()

	output_path = out_dir / f"{gesture}_{recording_name}.png"
	fig.savefig(output_path, dpi=150)
	return fig, output_path


def preview_rq(base_dir, rq, num_samples, save_only, rng):
	sample_filename = RQ_TO_FILE[rq]
	sample_files = collect_sample_files(base_dir, sample_filename)

	if not sample_files:
		print(f"No {sample_filename} files found under {base_dir}")
		return

	n = min(num_samples, len(sample_files))
	chosen = rng.sample(sample_files, k=n)
	out_dir = get_preview_dir(rq)

	print(f"\n{rq.upper()}: previewing {n}/{len(sample_files)} recordings")
	print(f"Saving previews to: {out_dir}")

	for gesture, recording_name, sample_file in chosen:
		sample_dict = np.load(sample_file, allow_pickle=True).item()
		fig, output_path = draw_sample_dict(
			gesture=gesture,
			recording_name=recording_name,
			sample_dict=sample_dict,
			rq=rq,
			out_dir=out_dir,
		)
		print(f"  saved: {output_path}")

		if save_only:
			plt.close(fig)

	if not save_only:
		plt.show()


def main():
	args = parse_args()
	base_dir = get_base_dir()
	rng = random.Random(args.seed)

	if args.num <= 0:
		raise ValueError("--num must be > 0")

	if args.rq == "all":
		rq_list = ["rq1", "rq2", "rq3"]
	else:
		rq_list = [args.rq]

	for rq in rq_list:
		preview_rq(
			base_dir=base_dir,
			rq=rq,
			num_samples=args.num,
			save_only=args.save_only,
			rng=rng,
		)


if __name__ == "__main__":
	main()
