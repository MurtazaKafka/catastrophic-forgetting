import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader

class EWC:
    """Elastic Weight Consolidation (EWC) implementation"""
    def __init__(self, model, dataset, device):
        """
        Args:
            model: Neural network model
            dataset: Dataset to compute Fisher Information
            device: Computing device (CPU/GPU)
        """
        self.model = copy.deepcopy(model)
        self.device = device
        self.params = {n: p.clone() for n, p in self.model.named_parameters()}
        self.fisher = self.compute_fisher_information(dataset)

    def compute_fisher_information(self, dataset):
        """Compute Fisher Information Matrix"""
        fisher = {n: torch.zeros_like(p, device=self.device) 
                 for n, p in self.model.named_parameters()}
        
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

        for n in fisher.keys():
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        """Compute EWC penalty"""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss