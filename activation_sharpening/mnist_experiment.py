import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

class ActivationSharpening(nn.Module):
    def __init__(self, alpha=5, threshold=0.3):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, x):
        sharpened = torch.pow(torch.abs(x), self.alpha) * torch.sign(x)
        return torch.nn.functional.softplus(sharpened - self.threshold) - torch.nn.functional.softplus(-sharpened - self.threshold)

class Network(nn.Module):
    def __init__(self, use_sharpening=False):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.use_sharpening = use_sharpening
        if use_sharpening:
            self.sharpening = ActivationSharpening(alpha=1.5, threshold=0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        if self.use_sharpening:
            x = self.sharpening(x)
        x = torch.relu(self.fc2(x))
        if self.use_sharpening:
            x = self.sharpening(x)
        x = self.fc3(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def run_experiment(use_sharpening):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    # Create two tasks: digits 0-4 and digits 5-9
    train_loader_task1 = DataLoader(Subset(mnist_train, [i for i in range(len(mnist_train)) if mnist_train.targets[i] < 5]), batch_size=64, shuffle=True)
    train_loader_task2 = DataLoader(Subset(mnist_train, [i for i in range(len(mnist_train)) if mnist_train.targets[i] >= 5]), batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)

    model = Network(use_sharpening=use_sharpening).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    accuracies = []

    # Train on task 1
    for epoch in range(5):
        train(model, train_loader_task1, optimizer, criterion, device)
        accuracy = test(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.4f}")

    # Train on task 2
    for epoch in range(5):
        train(model, train_loader_task2, optimizer, criterion, device)
        accuracy = test(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 6}, Accuracy: {accuracy:.4f}")

    return accuracies

# Run experiments
accuracies_without_sharpening = run_experiment(use_sharpening=False)
accuracies_with_sharpening = run_experiment(use_sharpening=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), accuracies_without_sharpening, label='Without Sharpening')
plt.plot(range(1, 11), accuracies_with_sharpening, label='With Sharpening')
plt.axvline(x=5.5, color='r', linestyle='--', label='Task Switch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over time with and without Activation Sharpening')
plt.legend()
plt.savefig('catastrophic_forgetting_experiment.png')
plt.close()

print("Experiment completed. Results saved in 'catastrophic_forgetting_experiment.png'")