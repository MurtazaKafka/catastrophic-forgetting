import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Permuted MNIST Task Generator
def permute_mnist(mnist_data, mnist_targets, seed):
    np.random.seed(seed)

    permute_idx = np.random.permutation(12 * 12)

    if mnist_data.dim() == 3:
        mnist_data = mnist_data.unsqueeze(1)

    N = mnist_data.shape[0]
    permuted_data = mnist_data.clone()

    center = permuted_data[:, :, 8:20, 8:20].clone()
    
    center = center.view(N, -1)[:, permute_idx].view(N, 1, 12, 12)

    permuted_data[:, :, 8:20, 8:20] = center

    return TensorDataset(permuted_data.float(), mnist_targets)

# Elastic Weight Consolidation (EWC)
class EWC:
    def __init__(self, model, dataset, device, fisher_n=100):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.fisher_n = fisher_n
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)
        
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.fisher_n, shuffle=True)
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            for n, p in self.model.named_parameters():
                precision_matrices[n] += p.grad.data ** 2 / len(self.dataset)
        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

# Training function with EWC regularization
def train_ewc(model, train_loader, optimizer, criterion, ewc=None, lam=100):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if ewc:
            loss += lam * ewc.penalty(model)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return test_loss, accuracy, precision, recall, f1

# Hyperparameters
batch_size = 100
epochs = 20
learning_rate = 0.001
lam = 100  # Regularization strength for EWC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

# Model and optimizer
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create permuted datasets for multiple tasks
num_tasks = 3
permuted_train_datasets = [permute_mnist(train_dataset.data, train_dataset.targets, seed=i) for i in range(num_tasks)]
permuted_test_datasets = [permute_mnist(test_dataset.data, test_dataset.targets, seed=i) for i in range(num_tasks)]

# Training the model on multiple permuted MNIST tasks with EWC
for task in range(num_tasks):
    print(f"Training on task {task + 1}")
    train_loader = DataLoader(permuted_train_datasets[task], batch_size=batch_size, shuffle=True)
    
    # Create an EWC object after each task
    if task > 0:
        ewc = EWC(model, permuted_train_datasets[task-1], device)
    else:
        ewc = None

    for epoch in range(epochs):
        train_loss = train_ewc(model, train_loader, optimizer, criterion, ewc, lam)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")

    # Test on all tasks after training on the current task
    print(f"\nTesting after task {task + 1}:")
    for test_task in range(task + 1):
        test_loader = DataLoader(permuted_test_datasets[test_task], batch_size=batch_size, shuffle=False)
        test_loss, accuracy, precision, recall, f1 = test(model, test_loader, device)
        print(f"Task {test_task + 1}: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print()

print("Training and testing completed.")



def run_ewc_experiment(num_tasks=3, epochs=20, use_ewc=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Store accuracies for each task over time
    task_accuracies = [[] for _ in range(num_tasks)]

    for task in range(num_tasks):
        print(f"Training on task {task + 1}")
        train_loader = DataLoader(permuted_train_datasets[task], batch_size=100, shuffle=True)
        
        ewc = EWC(model, permuted_train_datasets[task-1], device) if use_ewc and task > 0 else None

        for epoch in range(epochs):
            train_ewc(model, train_loader, optimizer, criterion, ewc, lam=100 if use_ewc else 0)

            # Test on all tasks seen so far
            for test_task in range(task + 1):
                test_loader = DataLoader(permuted_test_datasets[test_task], batch_size=100, shuffle=False)
                _, accuracy, _, _, _ = test(model, test_loader, device)
                task_accuracies[test_task].append(accuracy)

        # Pad accuracies for tasks not yet seen
        for future_task in range(task + 1, num_tasks):
            task_accuracies[future_task].append(0)  # or np.nan if you prefer

    return task_accuracies

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

# Run experiments
results_with_ewc = run_ewc_experiment(use_ewc=True)
results_without_ewc = run_ewc_experiment(use_ewc=False)

# Plot results
plot_results(results_with_ewc, results_without_ewc)