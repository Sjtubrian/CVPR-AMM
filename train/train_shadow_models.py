import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

shadow_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=train_transform
)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Split the CIFAR-10 training set into a target-model subset and a remaining pool.
target_train_set, remaining_train_set = random_split(trainset, [20000, 30000])

target_train_indices = target_train_set.indices
remaining_indices = list(set(range(len(trainset))) - set(target_train_indices))

os.makedirs("target_model", exist_ok=True)
os.makedirs("shadow_models_A_new", exist_ok=True)
os.makedirs("shadow_models_B_new", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class CustomImageDataset(Dataset):
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


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def adjust_learning_rate(optimizer, epoch, num_epochs, initial_lr):
    lr = initial_lr * (0.5 * (1 + math.cos(epoch * math.pi / num_epochs)))
    if epoch < 10:
        lr = lr * (epoch / 10)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_model(model, train_loader, epochs=100, initial_lr=0.1):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch, epochs, initial_lr)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader)}")

    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


# -----------------------------------------------------------------------------
# Optional target-model training block
# -----------------------------------------------------------------------------

'''
target_model = CIFAR10ResNet18()
target_train_loader = DataLoader(target_train_set, batch_size=128, shuffle=True, num_workers=4)
target_model = train_model(target_model, target_train_loader)
save_model(target_model, './target_model/target_model_cifar10.pth')
'''


# -----------------------------------------------------------------------------
# Optional fixed-index generation block
# -----------------------------------------------------------------------------

'''
fixed_shadow_indices_1 = random.sample(target_train_indices, 2000)
with open('fixed_cifar10_indices_1.pkl', 'wb') as f:
    pickle.dump(fixed_shadow_indices_1, f)

fixed_shadow_indices_2 = random.sample(range(len(testset)), 2000)
with open('fixed_cifar10_indices_2.pkl', 'wb') as f:
    pickle.dump(fixed_shadow_indices_2, f)

exit()
'''


# -----------------------------------------------------------------------------
# Train and save 100 shadow-model pairs
# -----------------------------------------------------------------------------

for i in range(100):
    with open("fixed_cifar10_indices_1.pkl", "rb") as f:
        fixed_shadow_indices_1 = pickle.load(f)
    with open("fixed_cifar10_indices_2.pkl", "rb") as f:
        fixed_shadow_indices_2 = pickle.load(f)

    # Shadow model A mixes member images, fixed test images, and fabricated samples.
    shadow_indices_remaining = random.sample(remaining_indices, 14000)
    shadow_train_indices_A = fixed_shadow_indices_1 + shadow_indices_remaining

    shadow_train_images_A = [trainset[j][0] for j in shadow_train_indices_A]
    shadow_train_labels_A = [trainset[j][1] for j in shadow_train_indices_A]

    shadow_test_images_A = [testset[j][0] for j in fixed_shadow_indices_2]
    shadow_test_labels_A = [testset[j][1] for j in fixed_shadow_indices_2]

    adversarial_samples_file = "./Max_adversarial_samples/cifar10/stable_adaptive_pgd_8.pt"
    adv_images, adv_labels = torch.load(adversarial_samples_file)
    augmented_adv_images = torch.stack([image for image in adv_images])
    adv_labels = adv_labels.tolist()

    all_images = shadow_train_images_A + shadow_test_images_A + [image for image in augmented_adv_images]
    all_labels = shadow_train_labels_A + shadow_test_labels_A + adv_labels

    combined_train_set = CustomImageDataset(all_images, all_labels, transform=shadow_transform)
    shadow_train_loader_A = DataLoader(combined_train_set, batch_size=128, shuffle=True, num_workers=4)

    shadow_model_A = CIFAR10ResNet18()
    shadow_model_A = train_model(shadow_model_A, shadow_train_loader_A)
    save_model(shadow_model_A, f"./shadow_models_A_new/cifar10_shadow_model_{i + 1}.pth")

    # Shadow model B is trained from the remaining clean pool only.
    shadow_train_indices_B = random.sample(remaining_indices, 20000)
    shadow_train_set_B = Subset(trainset, shadow_train_indices_B)
    shadow_train_loader_B = DataLoader(shadow_train_set_B, batch_size=128, shuffle=True, num_workers=4)

    shadow_model_B = CIFAR10ResNet18()
    shadow_model_B = train_model(shadow_model_B, shadow_train_loader_B)
    save_model(shadow_model_B, f"./shadow_models_B_new/cifar10_shadow_model_{i + 1}.pth")
