import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return t

class PermutedMNIST(datasets.MNIST):
    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        if permute_idx is None:
            permute_idx = torch.randperm(28 * 28)
        assert len(permute_idx) == 28 * 28
        
        # Store permuted data in a new attribute instead of modifying existing ones
        if self.train:
            self.permuted_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                       for img in self.data])
        else:
            self.permuted_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                      for img in self.data])

    def __getitem__(self, index):
        if self.train:
            img, target = self.permuted_data[index], self.targets[index]
        else:
            img, target = self.permuted_data[index], self.targets[index]
        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.permuted_data[sample_idx]]

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).sum().item()
    return correct / len(data_loader.dataset)

# Training script
def run_experiment(n_tasks=3, importance=10000, epochs=20):
    # Generate permutations
    permutations = [torch.randperm(784) for _ in range(n_tasks)]
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Store task accuracies
    task_accuracies_ewc = {i: [] for i in range(n_tasks)}
    task_accuracies_normal = {i: [] for i in range(n_tasks)}
    
    # Train with EWC
    for task_id in range(n_tasks):
        print(f"Training on task {task_id + 1}")
        
        # Create datasets for current task
        train_dataset = PermutedMNIST(train=True, permute_idx=permutations[task_id])
        test_dataset = PermutedMNIST(train=False, permute_idx=permutations[task_id])
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Get samples for Fisher computation
        fisher_samples = train_dataset.get_sample(200)
        
        # Create EWC object
        ewc = EWC(model, fisher_samples) if task_id > 0 else None
        
        # Training loop
        for epoch in range(epochs):
            if ewc is not None:
                ewc_train(model, optimizer, train_loader, ewc, importance)
            else:
                normal_train(model, optimizer, train_loader)
            
            # Test on all previous tasks
            for prev_task in range(task_id + 1):
                prev_test_dataset = PermutedMNIST(train=False, permute_idx=permutations[prev_task])
                prev_test_loader = DataLoader(prev_test_dataset, batch_size=128, shuffle=False)
                accuracy = test(model, prev_test_loader)
                task_accuracies_ewc[prev_task].append(accuracy)
    
    return task_accuracies_ewc

# Run experiments and plot results
if __name__ == "__main__":
    results = run_experiment(n_tasks=3, importance=10000, epochs=20)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for task_id, accuracies in results.items():
        plt.plot(accuracies, label=f'Task {task_id + 1}')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Test Accuracy')
    plt.title('EWC Performance on Permuted MNIST Tasks')
    plt.legend()
    plt.grid(True)
    plt.show()