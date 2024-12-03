import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.nn import functional as F
import torchvision

class EWCVisualizer:
    def __init__(self, model, train_loader, test_loader, num_tasks):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_tasks = num_tasks
        
    def plot_permuted_samples(self, num_samples=10):
        """
        Plot sample images from each permuted MNIST task
        """
        plt.figure(figsize=(15, 3 * self.num_tasks))
        
        for task_id in range(self.num_tasks):
            # Get samples from train loader
            images, _ = next(iter(self.train_loader[task_id]))
            images = images[:num_samples]
            
            for idx, img in enumerate(images):
                plt.subplot(self.num_tasks, num_samples, task_id * num_samples + idx + 1)
                plt.imshow(img.reshape(28, 28).cpu().numpy(), cmap='gray')
                plt.axis('off')
                if idx == 0:
                    plt.title(f'Task {task_id}', pad=20)
                    
        plt.tight_layout()
        plt.show()
        
    def plot_accuracy_comparison(self, standard_acc, ewc_acc):
        """
        Plot accuracy comparison between standard training and EWC
        for all tasks over time
        """
        plt.figure(figsize=(12, 6))
        
        # Plot settings
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_tasks))
        epochs_per_task = len(standard_acc[0]) // self.num_tasks
        x = np.arange(len(standard_acc[0]))
        
        # Plot standard training accuracies
        for task in range(self.num_tasks):
            plt.plot(x, standard_acc[task], '--', 
                    color=colors[task], 
                    label=f'Standard - Task {task}',
                    alpha=0.5)
            
        # Plot EWC accuracies
        for task in range(self.num_tasks):
            plt.plot(x, ewc_acc[task], 
                    color=colors[task], 
                    label=f'EWC - Task {task}')
            
        # Add task transition vertical lines
        for i in range(1, self.num_tasks):
            plt.axvline(x=i * epochs_per_task, 
                       color='gray', 
                       linestyle=':', 
                       alpha=0.5)
            plt.text(i * epochs_per_task, 0.2, 
                    f'Start Task {i}', 
                    rotation=90)
            
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: Standard vs EWC Training')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def get_predictions(self, task_id):
        """Helper function to get predictions for confusion matrix"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader[task_id]:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(inputs)
                preds = F.softmax(outputs, dim=1).max(dim=1)[1]
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                
        return np.array(all_preds), np.array(all_targets)
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all tasks to analyze catastrophic forgetting
        """
        fig, axes = plt.subplots(1, self.num_tasks, 
                                figsize=(5 * self.num_tasks, 5))
        
        if self.num_tasks == 1:
            axes = [axes]
            
        for task_id, ax in enumerate(axes):
            # Get predictions
            y_pred, y_true = self.get_predictions(task_id)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       ax=ax, square=True)
            ax.set_title(f'Task {task_id} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
        plt.tight_layout()
        plt.show()
        
    def plot_forgetting_analysis(self, standard_acc, ewc_acc):
        """
        Plot average accuracy and forgetting metrics for each task
        """
        plt.figure(figsize=(12, 5))
        
        # Calculate average accuracy for each task after training
        epochs_per_task = len(standard_acc[0]) // self.num_tasks
        std_final_acc = []
        ewc_final_acc = []
        
        for task in range(self.num_tasks):
            # Get accuracy at the end of training all tasks
            std_final_acc.append(standard_acc[task][-1])
            ewc_final_acc.append(ewc_acc[task][-1])
            
        # Plotting
        x = np.arange(self.num_tasks)
        width = 0.35
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, std_final_acc, width, label='Standard', alpha=0.7)
        plt.bar(x + width/2, ewc_final_acc, width, label='EWC', alpha=0.7)
        plt.xlabel('Task')
        plt.ylabel('Final Accuracy')
        plt.title('Final Accuracy Comparison')
        plt.xticks(x, [f'Task {i}' for i in range(self.num_tasks)])
        plt.legend()
        
        # Calculate forgetting (difference between max and final accuracy)
        std_forgetting = []
        ewc_forgetting = []
        
        for task in range(self.num_tasks):
            std_max = max(standard_acc[task][:epochs_per_task*(task+1)])
            std_forgetting.append(std_max - std_final_acc[task])
            
            ewc_max = max(ewc_acc[task][:epochs_per_task*(task+1)])
            ewc_forgetting.append(ewc_max - ewc_final_acc[task])
            
        plt.subplot(1, 2, 2)
        plt.bar(x - width/2, std_forgetting, width, label='Standard', alpha=0.7)
        plt.bar(x + width/2, ewc_forgetting, width, label='EWC', alpha=0.7)
        plt.xlabel('Task')
        plt.ylabel('Forgetting (Max - Final Accuracy)')
        plt.title('Catastrophic Forgetting Analysis')
        plt.xticks(x, [f'Task {i}' for i in range(self.num_tasks)])
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage:
"""
# Initialize visualizer
visualizer = EWCVisualizer(model, train_loader, test_loader, num_tasks=3)

# Plot permuted MNIST samples
visualizer.plot_permuted_samples()

# After training, plot comparisons
visualizer.plot_accuracy_comparison(standard_acc, ewc_acc)

# Plot confusion matrices
visualizer.plot_confusion_matrices()

# Plot forgetting analysis
visualizer.plot_forgetting_analysis(standard_acc, ewc_acc)
"""