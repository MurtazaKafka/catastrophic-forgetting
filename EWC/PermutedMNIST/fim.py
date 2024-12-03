import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict, Tuple
import random



class EWC:
    """
    Implementation of Elastic Weight Consolidation (EWC) for continual learning.
    Mathematical formulation:
    L(θ) = L_B(θ) + (λ/2)∑_i F_i(θ_i - θ*_A,i)²
    
    where:
    - L_B(θ): Loss for current task B
    - λ: Regularization strength
    - F_i: i-th diagonal element of the Fisher Information Matrix
    - θ_i: i-th parameter
    - θ*_A,i: i-th parameter value after training on task A
    """
    def __init__(self, model: nn.Module, dataset: torch.utils.data.Dataset, lambda_reg: float = 5000):
        self.model = model
        self.dataset = dataset
        self.lambda_reg = lambda_reg
        
        # Initialize stored parameters and Fisher Information Matrix
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher_dict = {}
        
    def compute_fisher(self, num_samples: int = 200):
        """
        Compute Fisher Information Matrix F = E[(∇_θ log p(y|x,θ))(∇_θ log p(y|x,θ))ᵀ]
        Uses empirical Fisher: F ≈ (1/N)∑ᵢ (∇_θ log p(yᵢ|xᵢ,θ))(∇_θ log p(yᵢ|xᵢ,θ))ᵀ
        """
        # Initialize Fisher Information for each parameter
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample from dataset
        sampler = torch.utils.data.RandomSampler(
            self.dataset, 
            replacement=True,
            num_samples=num_samples
        )
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=1
        )
        
        # Compute empirical Fisher Information Matrix
        for input_data, target in dataloader:
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(input_data)
            
            # Sample from output distribution (for classification tasks)
            if isinstance(output, torch.distributions.Distribution):
                sample = output.sample()
            else:
                # For standard classification, treat output as categorical distribution
                prob = F.softmax(output, dim=1)
                sample = torch.multinomial(prob, 1).squeeze()
            
            # Compute log probability
            log_prob = F.cross_entropy(output, sample.view(-1), reduction='sum')
            
            # Backward pass to get gradients
            log_prob.backward()
            
            # Accumulate squared gradients in Fisher Information Matrix
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad.data.pow(2) / num_samples
        
        self.fisher_dict = fisher_dict
        
    def ewc_loss(self, curr_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute EWC loss: L(θ) = L_B(θ) + (λ/2)∑_i F_i(θ_i - θ*_A,i)²
        
        Args:
            curr_loss: Current task loss L_B(θ)
        Returns:
            Total loss including EWC regularization
        """
        ewc_reg_loss = 0
        for n, p in self.model.named_parameters():
            # Skip if we don't have Fisher information for this parameter
            if n not in self.fisher_dict:
                continue
                
            # Compute EWC regularization term
            ewc_reg_loss += (self.fisher_dict[n] * (p - self.params[n]).pow(2)).sum()
        
        return curr_loss + (self.lambda_reg / 2) * ewc_reg_loss

    def update_fisher_params(self):
        """Update stored parameters after training on a task"""
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters()}













class PermutedMNIST(Dataset):
    """
    PermutedMNIST dataset where each task is a different permutation of MNIST pixels
    """
    def __init__(self, task_id: int, train: bool = True):
        """
        Args:
            task_id: ID of the task (determines the permutation)
            train: If True, use training set, else use test set
        """
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root='./data', 
            train=train, 
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Set random seed for reproducibility
        np.random.seed(task_id)
        
        # Generate permutation for this task
        self.permutation = torch.from_numpy(
            np.random.permutation(28 * 28)
        ).long()
        
        self.task_id = task_id
    
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.mnist[idx]
        
        # Flatten and permute image
        image = image.view(-1)[self.permutation].view(1, 28, 28)
        return image, label

class MLP(nn.Module):
    """
    Multi-layer perceptron for PermutedMNIST tasks
    """
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EWC:
    """
    Elastic Weight Consolidation
    """
    def __init__(self, model: nn.Module, lambda_reg: float = 5000):
        self.model = model
        self.lambda_reg = lambda_reg
        
        # Initialize dictionaries to store parameters and Fisher information
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        
    def compute_fisher(self, dataset: Dataset, num_samples: int = 200):
        """
        Compute Fisher Information Matrix
        """
        # Initialize Fisher information for each parameter
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dataloader for sampling
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=4
        )
        
        # Sample and compute Fisher information
        for i, (input_data, target) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            self.model.zero_grad()
            output = self.model(input_data)
            
            # Sample from output distribution
            prob = F.softmax(output, dim=1)
            sample = torch.multinomial(prob, 1).squeeze()
            
            # Compute log probability
            log_prob = F.cross_entropy(output, sample)
            log_prob.backward()
            
            # Accumulate Fisher information
            for n, p in self.model.named_parameters():
                fisher[n] += p.grad.data.pow(2) / num_samples
        
        # Update Fisher information
        if not self.fisher:
            self.fisher = fisher
        else:
            for n in self.fisher.keys():
                self.fisher[n] += fisher[n]
    
    def store_parameters(self):
        """Store current parameters"""
        self.params = {n: p.data.clone() for n, p in self.model.named_parameters()}
    
    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute EWC loss
        L(θ) = L_B(θ) + (λ/2)∑_i F_i(θ_i - θ*_A,i)²
        """
        ewc_loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                ewc_loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        
        return current_loss + (self.lambda_reg / 2) * ewc_loss

def evaluate(model: nn.Module, dataset: Dataset) -> float:
    """Evaluate model accuracy on dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4)
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return correct / total

def train_single_task(
    model: nn.Module, 
    dataset: Dataset,
    ewc: EWC = None,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.001
) -> List[float]:
    """
    Train model on a single task
    Returns list of losses during training
    """
    model.train()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            current_loss = F.cross_entropy(outputs, targets)
            
            # Compute total loss (including EWC penalty if applicable)
            loss = ewc.ewc_loss(current_loss) if ewc else current_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        losses.append(epoch_loss / len(dataloader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}')
    
    return losses

def run_permuted_mnist_experiment(
    num_tasks: int = 10,
    hidden_size: int = 256,
    lambda_reg: float = 5000,
    epochs_per_task: int = 5
):
    """Run complete PermutedMNIST experiment"""
    # Initialize model and EWC
    model = MLP(hidden_size=hidden_size)
    ewc = EWC(model, lambda_reg=lambda_reg)
    
    # Store accuracies for each task
    accuracies = np.zeros((num_tasks, num_tasks))
    
    # Train on each task sequentially
    for task_id in range(num_tasks):
        print(f"\nTraining on Task {task_id + 1}")
        
        # Create datasets for current task
        train_dataset = PermutedMNIST(task_id=task_id, train=True)
        
        # Train on current task
        train_single_task(
            model=model,
            dataset=train_dataset,
            ewc=ewc if task_id > 0 else None,
            epochs=epochs_per_task
        )
        
        # After training on current task:
        # 1. Compute Fisher information
        ewc.compute_fisher(train_dataset)
        # 2. Store parameters
        ewc.store_parameters()
        
        # Evaluate on all tasks seen so far
        for prev_task_id in range(task_id + 1):
            test_dataset = PermutedMNIST(task_id=prev_task_id, train=False)
            accuracy = evaluate(model, test_dataset)
            accuracies[task_id, prev_task_id] = accuracy
            print(f"Accuracy on task {prev_task_id + 1}: {accuracy:.4f}")
    
    return accuracies

def plot_results(accuracies: np.ndarray):
    """Plot results of the experiment"""
    plt.figure(figsize=(10, 8))
    plt.imshow(accuracies, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('Task Number')
    plt.ylabel('After Training Task')
    plt.title('Task Accuracy Matrix (EWC)')
    
    # Add text annotations
    for i in range(accuracies.shape[0]):
        for j in range(accuracies.shape[1]):
            if not np.isnan(accuracies[i, j]):
                plt.text(j, i, f'{accuracies[i, j]:.2f}', 
                        ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run experiment
    print("Starting PermutedMNIST experiment with EWC...")
    accuracies = run_permuted_mnist_experiment(
        num_tasks=10,
        hidden_size=256,
        lambda_reg=5000,
        epochs_per_task=5
    )
    
    # Plot results
    plot_results(accuracies)