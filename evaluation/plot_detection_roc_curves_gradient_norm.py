import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Create output folder
output_folder = 'Final_detection_figure_gradient_norm'
os.makedirs(output_folder, exist_ok=True)

plt.switch_backend('agg')  # Use 'agg' backend for file output
plt.figure(figsize=(10,8))  # Adjust figure size
plt.style.use('seaborn-v0_8-darkgrid')

plt.rc('font', family='Liberation Serif')
font_label = {'family': 'Liberation Serif', 'weight': 'normal', 'size': 23}

# Define the datasets and attacks
datasets = ['cifar10', 'cifar100', 'cinic', 'svhn']
attacks = ['fgsm', 'bim', 'pgd', 'cw', 'adaptive_pgd', 'stable_adaptive_pgd']

# Loop over datasets and attacks
for dataset in datasets:
    for attack in attacks:
        plt.figure(figsize=(10,8))  # Reset figure for each dataset and attack
        legend_labels = []

        # Load the original data
        original_shadow_loader_1 = np.load(f'./update_save_statistics/{dataset}/original_train_1_input.npy')
        native_index = np.load(f'./percentile_index/{dataset}/train_1_index.npy')
        original_shadow_loader_1 = original_shadow_loader_1[native_index]

        # Plot ROC curve for the attack
        for i in range(1, 9):  # Loop from 1 to 8 for the tail numbers
            adv_data = np.load(f'./update_save_statistics/{dataset}/{attack}_{i}_input.npy')
            adv_index = np.load(f'./percentile_index/{dataset}/{attack}_{i}_index.npy')
            adv_data = adv_data[adv_index]

            # Labels for each dataset
            labels_1 = np.ones(len(original_shadow_loader_1))  # Labels are 1
            labels_2 = np.zeros(len(adv_data))   # Labels are 0

            # Combine the datasets
            statistics_all = np.concatenate([original_shadow_loader_1, adv_data])
            labels_all = np.concatenate([labels_1, labels_2])

            # Calculate AUC score
            auc_score = roc_auc_score(labels_all, statistics_all)
            print(f"AUC for {attack}_{i} on {dataset}: {auc_score}")

            # Plot ROC curve
            fpr_all, tpr_all, thresholds_all = roc_curve(labels_all, statistics_all)
            plt.plot(fpr_all, tpr_all, lw=3, label=f'$\epsilon =$ {i}.0/255 (AUC = {auc_score:.4f})', alpha=0.8)
            legend_labels.append(f'$\epsilon =$ {i}.0/255 (AUC = {auc_score:.4f})')

        # Plot the diagonal (no skill)
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.3)

        # Set limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Add labels and title
        plt.xlabel('FALSE POSITIVE RATE', fontdict=font_label)
        plt.ylabel('TRUE POSITIVE RATE', fontdict=font_label)

        # Set font for ticks
        plt.yticks(fontsize=23, family='Liberation Serif')
        plt.xticks(fontsize=23, family='Liberation Serif')

        # Add legend in the lower right corner
        plt.legend(loc="lower right", fontsize=15)

        # Save the ROC curve plot as PNG and PDF
        png_filename = os.path.join(output_folder, f'roc_curve_{dataset}_{attack}_gradient_norm.png')
        pdf_filename = os.path.join(output_folder, f'roc_curve_{dataset}_{attack}_gradient_norm.pdf')
        plt.savefig(png_filename)
        plt.savefig(pdf_filename)
        plt.close()

        print(f"ROC curve for {dataset} with {attack} saved as '{png_filename}'")