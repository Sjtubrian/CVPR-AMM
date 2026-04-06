import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


plt.switch_backend("agg")
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-darkgrid")
plt.rc("font", family="Liberation Serif")
font2 = {"family": "Liberation Serif", "weight": "normal", "size": 25}

original_shadow_loader_1 = -np.load("./loss_p/cifar100/train_1.npy")
native_2_phi = -np.load("./loss_p/cifar100/test_1.npy")
fgsm_phi = -np.load("./loss_p/cifar100/fgsm_6.npy")
bim_phi = -np.load("./loss_p/cifar100/bim_6.npy")
pgd_phi = -np.load("./loss_p/cifar100/pgd_6.npy")
cw_phi = -np.load("./loss_p/cifar100/cw_6.npy")
adaptive_pgd_phi = -np.load("./loss_p/cifar100/adaptive_pgd_6.npy")
stable_adaptive_pgd_phi = -np.load("./loss_p/cifar100/stable_adaptive_pgd_6.npy")

labels_1 = np.ones(len(original_shadow_loader_1))
labels_2 = np.zeros(len(native_2_phi))


def plot_roc_curve(statistics, dataset_name, fpr_values=(0.1, 0.01, 0.05, 0.2)):
    statistics_all = np.concatenate([original_shadow_loader_1, statistics])
    labels_all = np.concatenate([labels_1, labels_2])

    auc_score = roc_auc_score(labels_all, statistics_all)
    print(f"AUC for {dataset_name}: {auc_score}")

    fpr, tpr, thresholds = roc_curve(labels_all, statistics_all)
    plt.plot(fpr, tpr, lw=3, label=f"{dataset_name} (AUC = {auc_score:.3f})", alpha=0.7)

    for target_fpr in fpr_values:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f"Threshold at FPR = {target_fpr} for {dataset_name}: {thresholds[idx]}")


for statistics, name in [
    (native_2_phi, "Natural"),
    (fgsm_phi, "I-FGSM"),
    (bim_phi, "I-BIM"),
    (pgd_phi, "I-PGD"),
    (cw_phi, "I-CW"),
    (adaptive_pgd_phi, "I-APGD"),
    (stable_adaptive_pgd_phi, "OURS"),
]:
    plot_roc_curve(statistics, name)

plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", alpha=0.7)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("FPR", fontdict=font2)
plt.ylabel("TPR", fontdict=font2)
plt.legend(loc="upper left", fontsize=16)
plt.xticks(fontsize=24, family="Liberation Serif")
plt.yticks(fontsize=24, family="Liberation Serif")
plt.savefig("roc_curve_comparison_loss_cifar100_4.png", dpi=300)
plt.close()

print("ROC curve comparison saved as 'roc_curve_comparison_loss_cifar100_4.png'")
