import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
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

with open("fixed_cifar10_indices_1.pkl", "rb") as f:
    fixed_shadow_indices_1 = pickle.load(f)
with open("fixed_cifar10_indices_2.pkl", "rb") as f:
    fixed_shadow_indices_2 = pickle.load(f)

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

fixed_shadow_set_1 = Subset(trainset, fixed_shadow_indices_1)
fixed_shadow_set_2 = Subset(testset, fixed_shadow_indices_2)

batch_size = 100
fixed_shadow_loader_1 = DataLoader(fixed_shadow_set_1, batch_size=batch_size, shuffle=False)
fixed_shadow_loader_2 = DataLoader(fixed_shadow_set_2, batch_size=batch_size, shuffle=False)

os.makedirs("Max_adversarial_samples/cifar10", exist_ok=True)
os.makedirs("loss_p/cifar10", exist_ok=True)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


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


model = CIFAR10ResNet18()
model.load_state_dict(torch.load("./target_model/target_model_cifar10.pth"))
model.to(device)
model.eval()


# -----------------------------------------------------------------------------
# Attack implementations
# -----------------------------------------------------------------------------

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    attack_images = images - epsilon * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    return attack_images


def bim_attack(model, images, labels, epsilon, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


def pgd_attack(model, images, labels, epsilon, alpha, iters, random_start=True):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    if random_start:
        images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        images = torch.clamp(images, 0, 1)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


def cwloss(logits, labels):
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).float()
    correct_logit = torch.sum(one_hot_labels * logits, dim=1)
    max_wrong_logit = torch.max((1 - one_hot_labels) * logits - one_hot_labels * 1e9, dim=1)[0]
    loss = torch.mean(max_wrong_logit - correct_logit)
    return loss


def cw_attack(model, images, labels, epsilon, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = cwloss(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


def adaptive_pgd_attack(model, images, labels, epsilon, iters, initial_alpha, decay_factor=0.9, adaptive=True):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    alpha = initial_alpha

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        if adaptive:
            alpha = alpha * decay_factor

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


def simple_adaptive_pgd_attack(model, images, labels, epsilon, iters, initial_alpha, decay_factor=0.9, adaptive=True):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    alpha = initial_alpha

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        if adaptive:
            alpha = initial_alpha * (1 + torch.cos(torch.tensor(torch.pi) * torch.tensor(i) / torch.tensor(iters))) / 2

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


def stable_adaptive_pgd_attack(model, images, labels, epsilon, iters, initial_alpha, decay_factor=0.9, momentum_factor=0.75, adaptive=True):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    alpha = initial_alpha
    momentum = torch.zeros_like(images).to(device)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        if adaptive:
            alpha = initial_alpha * (1 + torch.cos(torch.tensor(torch.pi) * torch.tensor(i) / torch.tensor(iters))) / 2

        momentum = momentum_factor * momentum + (1 - momentum_factor) * images.grad
        adv_images = images - alpha * torch.sign(momentum)
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images


# -----------------------------------------------------------------------------
# Saving utilities
# -----------------------------------------------------------------------------

def calculate_accuracy_and_loss(model, adv_images, labels):
    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels, reduction="mean")
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy, loss.item()


def save_adversarial_samples(attack_name, loader, attack_id, attack_func, *args):
    all_adv_images = []
    all_labels = []
    all_loss_values = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = attack_func(model, images, labels, *args)
        all_adv_images.append(adv_images.cpu().detach())
        all_labels.append(labels.cpu().detach())

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels, reduction="none")
        all_loss_values.extend(loss.cpu().detach().numpy())
        total_loss += loss.sum().item()
        total_samples += labels.size(0)

    all_adv_images = torch.cat(all_adv_images)
    all_labels = torch.cat(all_labels)

    torch.save((all_adv_images, all_labels), f"Max_adversarial_samples/{attack_name}_{attack_id}.pt")
    np.save(f"./loss_p/{attack_name}_{attack_id}.npy", np.array(all_loss_values))

    print(np.mean(np.array(all_loss_values)))

    average_loss = total_loss / total_samples
    accuracy = calculate_accuracy_and_loss(model, all_adv_images.to(device), all_labels.to(device))[0]
    print(f"{attack_name} Attack: Accuracy = {accuracy * 100:.4f}%, Average Loss = {average_loss:.6f}")


def calculate_original_accuracy_and_loss(loader):
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    accuracy = correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


# -----------------------------------------------------------------------------
# Run the generation pipeline
# -----------------------------------------------------------------------------

attack_id = 1

original_accuracy_1, original_loss_1 = calculate_original_accuracy_and_loss(fixed_shadow_loader_1)
print(f"Original Accuracy on fixed_shadow_set_1: {original_accuracy_1 * 100:.2f}%, Loss: {original_loss_1:.4f}")

original_accuracy_2, original_loss_2 = calculate_original_accuracy_and_loss(fixed_shadow_loader_2)
print(f"Original Accuracy on fixed_shadow_set_2: {original_accuracy_2 * 100:.2f}%, Loss: {original_loss_2:.4f}")

save_adversarial_samples("cifar10/train", fixed_shadow_loader_1, attack_id, fgsm_attack, 0)
save_adversarial_samples("cifar10/test", fixed_shadow_loader_2, attack_id, fgsm_attack, 0)

for attack_id in range(1, 9):
    epsilon_all = attack_id / 255

    save_adversarial_samples("cifar10/fgsm", fixed_shadow_loader_2, attack_id, fgsm_attack, epsilon_all)
    save_adversarial_samples("cifar10/bim", fixed_shadow_loader_2, attack_id, bim_attack, epsilon_all, epsilon_all / 4, 100)
    save_adversarial_samples("cifar10/pgd", fixed_shadow_loader_2, attack_id, pgd_attack, epsilon_all, epsilon_all / 4, 100)
    save_adversarial_samples("cifar10/cw", fixed_shadow_loader_2, attack_id, cw_attack, epsilon_all, epsilon_all / 4, 100)
    save_adversarial_samples("cifar10/adaptive_pgd", fixed_shadow_loader_2, attack_id, adaptive_pgd_attack, epsilon_all, 100, epsilon_all / 4)
    save_adversarial_samples("cifar10/simple_adaptive_pgd", fixed_shadow_loader_2, attack_id, simple_adaptive_pgd_attack, epsilon_all, 100, epsilon_all / 4)
    save_adversarial_samples("cifar10/stable_adaptive_pgd", fixed_shadow_loader_2, attack_id, stable_adaptive_pgd_attack, epsilon_all, 100, epsilon_all / 4)
