#!/usr/bin/env python3
"""
count_tokens.py - Count total tokens in your dataset
"""

import numpy as np
from pathlib import Path

def main():
    data_dir = "/fs/scratch/PAS2836/mugezhang/ipa_gpt_data/tokenized_bin_headered/headered_tokenized_bin_eng_spa_normal"
    train_files = sorted(Path(data_dir).glob("data_train_*.bin"))

    if not train_files:
        print(f"No training files found in {data_dir}")
        return

    total_tokens = 0
    for file in train_files:
        with open(file, 'rb') as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            num_tokens = header[2]
            total_tokens += num_tokens
            print(f"{file.name}: {num_tokens:,} tokens")

    print(f"\n{'=' * 60}")
    print(f"Total training tokens: {total_tokens:,}")
    print(f"Tokens per iteration (8 GPUs): 524,288")
    print(f"Iterations for 1 epoch: {total_tokens / 524288:.0f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()