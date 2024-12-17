import random
import torch
from torchvision import datasets


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