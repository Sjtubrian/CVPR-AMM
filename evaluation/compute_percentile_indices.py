import os
import random

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET = "svhn"
ATTACKS = [
    "fgsm",
    "bim",
    "pgd",
    "cw",
    "adaptive_pgd",
    "simple_adaptive_pgd",
    "stable_adaptive_pgd",
]

loss_test = np.load(f"./loss_p/{DATASET}/test_1.npy")
loss_train = np.load(f"./loss_p/{DATASET}/train_1.npy")

percentile_10 = np.percentile(loss_test, 10)
count_at_or_below_percentile = np.sum(loss_test <= percentile_10)

print(f"Number of samples at or below the 10th percentile: {count_at_or_below_percentile}")
print(f"10th percentile value: {percentile_10}")

index_save_path = f"./percentile_index/{DATASET}/"
os.makedirs(index_save_path, exist_ok=True)


# -----------------------------------------------------------------------------
# Save attack-wise indices
# -----------------------------------------------------------------------------

for attack in ATTACKS:
    for attack_num in range(1, 9):
        adv_loss = np.load(f"./loss_p/{DATASET}/{attack}_{attack_num}.npy")
        indices_less_than_percentile = np.where(adv_loss < percentile_10)[0]
        print(len(indices_less_than_percentile))

        index_filename = f"{attack}_{attack_num}_index.npy"
        np.save(os.path.join(index_save_path, index_filename), indices_less_than_percentile)
        print(f"Saved indices for {attack}_{attack_num} to {index_filename}")


# -----------------------------------------------------------------------------
# Save train/test reference indices
# -----------------------------------------------------------------------------

indices_less_than_percentile = np.where(loss_train < percentile_10)[0]
print(len(indices_less_than_percentile))
np.save(os.path.join(index_save_path, "train_1_index.npy"), indices_less_than_percentile)
print("Saved indices for train_1_index")

indices_less_than_percentile = np.where(loss_test < percentile_10)[0]
print(len(indices_less_than_percentile))
np.save(os.path.join(index_save_path, "test_1_index.npy"), indices_less_than_percentile)
print("Saved indices for test_1_index")

