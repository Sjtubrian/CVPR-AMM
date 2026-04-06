import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

with open("fixed_cifar10_indices_1.pkl", "rb") as f:
    fixed_shadow_indices_1 = pickle.load(f)
with open("fixed_cifar10_indices_2.pkl", "rb") as f:
    fixed_shadow_indices_2 = pickle.load(f)

fixed_shadow_set = Subset(trainset, fixed_shadow_indices_1)
fixed_shadow_loader = DataLoader(fixed_shadow_set, batch_size=1, shuffle=False, num_workers=4)

fixed_test_set = Subset(testset, fixed_shadow_indices_2)
test_loader = DataLoader(fixed_test_set, batch_size=1, shuffle=False, num_workers=4)

os.makedirs("save_members_new", exist_ok=True)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


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


shadow_models_A = []
for i in range(100):
    model = CIFAR10ResNet18()
    model.load_state_dict(torch.load(f"./shadow_models_A_new/cifar10_shadow_model_{i + 1}.pth"))
    model.eval()
    shadow_models_A.append(model)

shadow_models_B = []
for i in range(100):
    model = CIFAR10ResNet18()
    model.load_state_dict(torch.load(f"./shadow_models_B_new/cifar10_shadow_model_{i + 1}.pth"))
    model.eval()
    shadow_models_B.append(model)

target_model = CIFAR10ResNet18()
target_model.load_state_dict(torch.load("./target_model/target_model_cifar10.pth"))
target_model.eval()


# -----------------------------------------------------------------------------
# Confidence export
# -----------------------------------------------------------------------------

def calculate_p_y(models, inputs, labels):
    p_y = []
    for model in models:
        model.to(inputs.device)
        with torch.no_grad():
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            p_y.append(probs.gather(1, labels.view(-1, 1)).squeeze().item())
    return p_y


def save_data(loader, models_A, models_B, target_model, label, filename):
    data_list = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        p_y_A = calculate_p_y(models_A, inputs, labels)
        p_y_B = calculate_p_y(models_B, inputs, labels)
        target_p_y = calculate_p_y([target_model], inputs, labels)[0]

        data_list.append((p_y_A, p_y_B, target_p_y, label))

    with open(filename, "wb") as f:
        pickle.dump(data_list, f)


class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


adversarial_samples_file = "./Max_adversarial_samples/cifar10/stable_adaptive_pgd_8.pt"
adv_images, adv_labels = torch.load(adversarial_samples_file)
adversarial_dataset = AdversarialDataset(adv_images, adv_labels, transform=None)
adversarial_loader = DataLoader(adversarial_dataset, batch_size=1, shuffle=False, num_workers=4)

save_data(adversarial_loader, shadow_models_A, shadow_models_B, target_model, 0, "./save_members_new/adversarial_data_cifar10.pkl")
save_data(fixed_shadow_loader, shadow_models_A, shadow_models_B, target_model, 1, "./save_members_new/member_data_cifar10.pkl")
save_data(test_loader, shadow_models_A, shadow_models_B, target_model, 0, "./save_members_new/non_member_data_cifar10.pkl")
