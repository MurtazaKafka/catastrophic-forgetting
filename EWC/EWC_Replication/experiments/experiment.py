import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Permuted MNIST Task Generator
def permute_mnist(mnist_data, mnist_targets, seed):
    np.random.seed(seed)

    permute_idx = np.random.permutation(12 * 12)

    if (mnist_data.dim() == 3):
        mnist_data = mnist_data.unsqueeze(1)

    N = mnist_data.shape[0]
    permuted_data = mnist_data.clone()

    center = permuted_data[:, :, 8:20, 8:20].clone()
    
    center = center.view(N, -1)[:, permute_idx].view(N, 1, 12, 12)

    permuted_data[:, :, 8:20, 8:20] = center

    return TensorDataset(permuted_data.float(), mnist_targets)

# Elastic Weight Consolidation (EWC)
class EWC:
    def __init__(self, model, dataset, device):
        self.model = copy.deepcopy(model)
        self.device = device
        self.model.eval()

        # Store parameters
        self.params = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Compute Fisher Information Matrix
        self.fisher = self.compute_fisher_information(dataset)

    def compute_fisher_information(self, dataset):
        # Initialize Fisher Information Matrix
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters()}

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)

        # Average Fisher Information
        for n in fisher.keys():
            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher = self.fisher[n]
                loss += (fisher * (p - self.params[n]).pow(2)).sum()
        return loss

# Training function with EWC regularization
def train(model, train_loader, optimizer, criterion, ewc=None, lam=0):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if ewc is not None:
            loss += lam * ewc.penalty(model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, preds = torch.max(output, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return test_loss / len(test_loader), accuracy, None, None, None  # Adjust as needed

# Hyperparameters
batch_size = 256
epochs = 10
learning_rate = 0.1
lam = 400  # Regularization strength for EWC
num_tasks = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

# Model and optimizer
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create permuted datasets
num_tasks = 3  # Number of tasks
permuted_train_datasets = []
permuted_test_datasets = []
for task in range(num_tasks):
    permutation = torch.randperm(28 * 28)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))
    ])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
    permuted_train_datasets.append(train_dataset)
    permuted_test_datasets.append(test_dataset)

# Initialize variables to store accuracies
accuracies_with_ewc = [[] for _ in range(num_tasks)]
accuracies_without_ewc = [[] for _ in range(num_tasks)]

# Training loop
for task in range(num_tasks):
    print(f"Training on task {task + 1}")
    train_loader = DataLoader(permuted_train_datasets[task], batch_size=batch_size, shuffle=True)
    test_loaders = [DataLoader(permuted_test_datasets[i], batch_size=batch_size, shuffle=False) for i in range(task + 1)]

    # Initialize EWC after the first task
    if task > 0:
        ewc = EWC(model, permuted_train_datasets[task - 1], device)
    else:
        ewc = None

    # Train the model
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, ewc, lam)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

    # Test on all tasks seen so far
    print(f"\nTesting after task {task + 1}:")
    for i, test_loader in enumerate(test_loaders):
        test_loss, accuracy, _, _, _ = test(model, test_loader, device)
        print(f"Task {i + 1}: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        accuracies_with_ewc[i].append(accuracy)

# Reset the model and optimizer
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop without EWC
# Initialize list to store accuracies
accuracies_without_ewc = []

for task in range(num_tasks):
    print(f"Training on task {task + 1} without EWC")
    train_loader = DataLoader(permuted_train_datasets[task], batch_size=batch_size, shuffle=True)
    test_loaders = [DataLoader(permuted_test_datasets[i], batch_size=batch_size, shuffle=False) for i in range(task + 1)]

    # Train the model
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

    # After training, test the model on all seen tasks
    total_accuracy = 0
    for test_task in range(task + 1):
        test_loader = DataLoader(permuted_test_datasets[test_task], batch_size=batch_size, shuffle=False)
        test_loss, accuracy, _, _, _ = test(model, test_loader, device)
        total_accuracy += accuracy
        print(f"Test on Task {test_task + 1}: Accuracy: {accuracy:.4f}")
    average_accuracy = total_accuracy / (task + 1)
    accuracies_without_ewc.append(average_accuracy)

def run_ewc_experiment(num_tasks=3, epochs=10, use_ewc=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model and optimizer
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize list to store average accuracies
    average_accuracies = []

    for task in range(num_tasks):
        print(f"Training on task {task + 1}{' with EWC' if use_ewc else ' without EWC'}")
        train_loader = DataLoader(permuted_train_datasets[task], batch_size=batch_size, shuffle=True)

        # Initialize EWC after the first task
        if use_ewc and task > 0:
            ewc = EWC(model, permuted_train_datasets[task - 1], device)
            current_lam = lam
        else:
            ewc = None
            current_lam = 0

        # Train the model
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion, ewc, current_lam)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

        # Test on all tasks seen so far
        total_accuracy = 0
        for test_task in range(task + 1):
            test_loader = DataLoader(permuted_test_datasets[test_task], batch_size=batch_size, shuffle=False)
            test_loss, accuracy, _, _, _ = test(model, test_loader, device)
            print(f"Test on Task {test_task + 1}: Accuracy: {accuracy:.4f}")
            total_accuracy += accuracy
        average_accuracy = total_accuracy / (task + 1)
        average_accuracies.append(average_accuracy)
        print()

    return average_accuracies

def plot_results(with_ewc, without_ewc):
    plt.figure(figsize=(10, 6))
    
    for i in range(len(with_ewc)):
        plt.plot(with_ewc[i], label=f'Task {i+1} (with EWC)', linestyle='-')
        plt.plot(without_ewc[i], label=f'Task {i+1} (without EWC)', linestyle='--')
    
    plt.xlabel('Number of tasks seen')
    plt.ylabel('Test accuracy')
    plt.title('EWC vs No EWC on Permuted MNIST Tasks')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracies(accuracies_with_ewc, accuracies_without_ewc):
    tasks = range(1, len(accuracies_with_ewc) + 1)
    plt.plot(tasks, accuracies_with_ewc, label='With EWC', marker='o')
    plt.plot(tasks, accuracies_without_ewc, label='Without EWC', marker='x')
    plt.xlabel('Task Number')
    plt.ylabel('Average Accuracy')
    plt.title('EWC vs. No EWC on Permuted MNIST')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run experiments
results_with_ewc = run_ewc_experiment(num_tasks=num_tasks, epochs=epochs, use_ewc=True)
results_without_ewc = run_ewc_experiment(num_tasks=num_tasks, epochs=epochs, use_ewc=False)

# Plot results
plot_accuracies(results_with_ewc, results_without_ewc)