import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

with open("fixed_cifar10_indices_1.pkl", "rb") as f:
    fixed_shadow_indices_1 = pickle.load(f)

with open("fixed_cifar10_indices_2.pkl", "rb") as f:
    fixed_shadow_indices_2 = pickle.load(f)

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

fixed_shadow_set_1 = Subset(trainset, fixed_shadow_indices_1)
fixed_shadow_set_2 = Subset(testset, fixed_shadow_indices_2)

batch_size = 1
fixed_shadow_loader_1 = DataLoader(fixed_shadow_set_1, batch_size=batch_size, shuffle=False)
fixed_shadow_loader_2 = DataLoader(fixed_shadow_set_2, batch_size=batch_size, shuffle=False)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class CIFAR10ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 10)

    def forward(self, x):
        return self.resnet18(x)


# Choose the GPU index used in your original environment if available.
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = CIFAR10ResNet18()
model.load_state_dict(torch.load("./target_model/target_model_cifar10.pth"))
model.to(device)
model.eval()


# -----------------------------------------------------------------------------
# Gradient-norm extraction
# -----------------------------------------------------------------------------

def compute_input_gradient_norm(images, labels, model):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    return images.grad.abs().sum().item()


def extract_gradient_norms(loader, save_path):
    gradient_norms = []

    for images, labels in loader:
        gradient_norm = compute_input_gradient_norm(images, labels, model)
        gradient_norms.append(gradient_norm)

    gradient_norms = np.array(gradient_norms)

    print(f"Max gradient norm: {np.max(gradient_norms):.6f}")
    print(f"Min gradient norm: {np.min(gradient_norms):.6f}")
    print(f"Median gradient norm: {np.median(gradient_norms):.6f}")
    print(f"Mean gradient norm: {np.mean(gradient_norms):.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, gradient_norms)


if __name__ == "__main__":
    print("Extracting gradient norms for training members...")
    extract_gradient_norms(
        fixed_shadow_loader_1,
        "./update_save_statistics/cifar10/original_train_1_input.npy",
    )

    print("Extracting gradient norms for non-members...")
    extract_gradient_norms(
        fixed_shadow_loader_2,
        "./update_save_statistics/cifar10/original_train_2_input.npy",
    )
