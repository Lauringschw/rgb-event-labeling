from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR = Path(os.getenv("SLIDING_DIR"))  # Seagate - read batches
SLIDING_DIR_T7 = Path(os.getenv("SLIDING_DIR_T7"))  # T7 - write output
SLIDING_DIR_T7.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 6

def merge_seagate_to_t7():
    """Read batches from Seagate, write merged file to T7."""
    
    print("="*60)
    print("MERGE: Seagate → T7")
    print("="*60)
    print(f"Reading from: {SLIDING_DIR}")
    print(f"Writing to:   {SLIDING_DIR_T7}\n")
    
    # Find batches on Seagate
    print("Finding batch files on Seagate...")
    batch_data_files = list(SLIDING_DIR.glob("histogram_data_batch_*.npy"))
    batch_label_files = list(SLIDING_DIR.glob("histogram_labels_batch_*.npy"))
    
    def get_batch_num(filepath):
        return int(filepath.stem.split('_')[-1])
    
    batch_data_files = sorted(batch_data_files, key=get_batch_num)
    batch_label_files = sorted(batch_label_files, key=get_batch_num)
    
    if len(batch_data_files) == 0:
        print("No batches found!")
        return
    
    print(f"Found {len(batch_data_files)} batches")
    print(f"Batch range: {get_batch_num(batch_data_files[0])} to {get_batch_num(batch_data_files[-1])}\n")
    
    # Count samples
    print("Counting samples...")
    total_samples = 0
    for label_file in batch_label_files:
        labels = np.load(label_file)
        total_samples += len(labels)
    
    # Get shape
    first_batch = np.load(batch_data_files[0])
    sample_shape = first_batch.shape[1:]
    del first_batch
    
    # Calculate storage
    bytes_needed = total_samples * 2 * 720 * 1280 * 4
    gb_needed = bytes_needed / 1e9
    
    # Check T7 space
    stat = os.statvfs(SLIDING_DIR_T7)
    available_gb = (stat.f_bavail * stat.f_frsize) / 1e9
    
    print(f"\nStorage check:")
    print(f"  Samples: {total_samples:,}")
    print(f"  Shape: {sample_shape}")
    print(f"  Size needed: {gb_needed:.1f} GB")
    print(f"  T7 available: {available_gb:.1f} GB")
    
    if gb_needed > available_gb * 0.95:
        print(f"\n❌ NOT ENOUGH SPACE on T7!")
        print(f"Need {gb_needed:.1f} GB, only have {available_gb:.1f} GB")
        print("\nMove rock recordings to Seagate first:")
        print("  mv /media/lau/T7/thesis/recordings/trial2/rock /media/lau/seagate/temp_rock/")
        return
    
    print(f"  ✓ Enough space!\n")
    
    # Create output on T7
    output_data = SLIDING_DIR_T7 / "histogram_data.npy"
    output_labels = SLIDING_DIR_T7 / "histogram_labels.npy"
    
    print("Creating memory-mapped files on T7 SSD...")
    data_memmap = np.lib.format.open_memmap(
        str(output_data), 
        mode='w+', 
        dtype=np.float32,
        shape=(total_samples,) + sample_shape
    )
    
    labels_memmap = np.lib.format.open_memmap(
        str(output_labels),
        mode='w+',
        dtype=np.int64,
        shape=(total_samples,)
    )
    
    print("✓ Files created!\n")
    
    # Merge
    current_idx = 0
    num_chunks = (len(batch_data_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print("Merging batches from Seagate to T7...\n")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(batch_data_files))
        
        print(f"Chunk {chunk_idx + 1}/{num_chunks}:")
        
        for i in range(start_idx, end_idx):
            batch_num = get_batch_num(batch_data_files[i])
            print(f"  [{i+1}/{len(batch_data_files)}] batch_{batch_num}...", end='', flush=True)
            
            batch_data = np.load(batch_data_files[i])
            batch_labels = np.load(batch_label_files[i])
            
            batch_size = len(batch_data)
            
            data_memmap[current_idx:current_idx + batch_size] = batch_data
            labels_memmap[current_idx:current_idx + batch_size] = batch_labels
            
            current_idx += batch_size
            
            print(f" ✓ ({current_idx:,}/{total_samples:,})")
            
            del batch_data
            del batch_labels
        
        print()
    
    # Flush
    print("Flushing to disk...")
    del data_memmap
    del labels_memmap
    
    # Verify
    print("\nVerifying output on T7...")
    final_labels = np.load(output_labels)
    
    rock_count = np.sum(final_labels == 0)
    paper_count = np.sum(final_labels == 1)
    scissor_count = np.sum(final_labels == 2)
    
    print(f"\nFinal dataset on T7:")
    print(f"  Rock: {rock_count:,}")
    print(f"  Paper: {paper_count:,}")
    print(f"  Scissor: {scissor_count:,}")
    print(f"  TOTAL: {len(final_labels):,}")
    print(f"  Size: {output_data.stat().st_size / 1e9:.2f} GB")
    
    print(f"\n✓ Merged files saved to: {SLIDING_DIR_T7}")
    print(f"✓ Batch files kept on Seagate as backup: {SLIDING_DIR}")
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update train_histogram.py to use SLIDING_DIR_T7")
    print("2. Run training on T7 (fast SSD)")
    print("3. (Optional) Delete Seagate batches to free space later")

if __name__ == "__main__":
    merge_seagate_to_t7()