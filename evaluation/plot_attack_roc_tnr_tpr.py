import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


output_folder = "Final_attack_figure"
os.makedirs(output_folder, exist_ok=True)

plt.switch_backend("agg")
plt.style.use("seaborn-v0_8-darkgrid")
plt.rc("font", family="Liberation Serif")
font_label = {"family": "Liberation Serif", "weight": "normal", "size": 23}

datasets_list = ["cifar10", "cifar100", "svhn", "cinic"]
tail_numbers = range(1, 9)


def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    return fpr[eer_index]


def plot_roc_curve(statistics, original_shadow_loader, dataset_name, txt_file):
    statistics_all = np.concatenate([original_shadow_loader, statistics])
    labels_all = np.concatenate([np.ones(len(original_shadow_loader)), np.zeros(len(statistics))])

    auc_score = roc_auc_score(labels_all, statistics_all)
    fpr, tpr, _ = roc_curve(labels_all, statistics_all)
    tnr = 1 - fpr
    plt.plot(tnr, tpr, lw=3, label=dataset_name, alpha=0.8)

    eer = calculate_eer(fpr, tpr)
    txt_file.write(f"{dataset_name} - EER: {eer:.6f}\n")
    txt_file.write(f"{dataset_name} - 1-AUC: {1 - auc_score:.6f}\n")


txt_filename = os.path.join(output_folder, "roc_curve_comparison_loss_all_1_auc.txt")
with open(txt_filename, "w") as txt_file:
    for dataset in datasets_list:
        for tail in tail_numbers:
            original_shadow_loader = -np.load(f"./loss_p/{dataset}/train_1.npy")
            native_phi = -np.load(f"./loss_p/{dataset}/test_1.npy")
            fgsm_phi = -np.load(f"./loss_p/{dataset}/fgsm_{tail}.npy")
            bim_phi = -np.load(f"./loss_p/{dataset}/bim_{tail}.npy")
            pgd_phi = -np.load(f"./loss_p/{dataset}/pgd_{tail}.npy")
            cw_phi = -np.load(f"./loss_p/{dataset}/cw_{tail}.npy")
            adaptive_pgd_phi = -np.load(f"./loss_p/{dataset}/adaptive_pgd_{tail}.npy")
            stable_adaptive_pgd_phi = -np.load(f"./loss_p/{dataset}/stable_adaptive_pgd_{tail}.npy")

            txt_file.write(f"\nDataset: {dataset}, Tail: {tail}\n")

            plot_datasets = [
                (native_phi, "Natural"),
                (fgsm_phi, "I-FGSM"),
                (bim_phi, "I-BIM"),
                (pgd_phi, "I-PGD"),
                (cw_phi, "I-CW"),
                (adaptive_pgd_phi, "I-APGD"),
                (stable_adaptive_pgd_phi, "OURS"),
            ]

            plt.figure(figsize=(10, 8))
            for data, label in plot_datasets:
                plot_roc_curve(data, original_shadow_loader, label, txt_file)

            t = np.linspace(1e-4, 1, 100)
            plt.plot(t, t, color="navy", lw=1.5, linestyle="--", alpha=0.3)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(0.82e-2, 1.2)
            plt.ylim(0.82e-2, 1.2)
            plt.xlabel("TRUE NEGATIVE RATE", fontdict=font_label)
            plt.ylabel("TRUE POSITIVE RATE", fontdict=font_label)
            plt.legend(loc="lower left", fontsize=15)
            plt.xticks(fontsize=23, family="Liberation Serif")
            plt.yticks(fontsize=23, family="Liberation Serif")

            png_filename = os.path.join(output_folder, f"roc_curve_comparison_loss_{dataset}_{tail}.png")
            pdf_filename = os.path.join(output_folder, f"roc_curve_comparison_loss_{dataset}_{tail}.pdf")
            plt.savefig(png_filename)
            plt.savefig(pdf_filename)
            plt.close()

            print(f"Saved ROC curves for dataset {dataset}, tail number {tail} as PNG and PDF.")
