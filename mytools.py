import numpy as np
import matplotlib.pyplot as plt

def count_class_distribution(loader, num_classes=100):
    class_counts = np.zeros(num_classes)
    
    for _, targets in loader:
        targets = targets.numpy()
        for target in targets:
            class_counts[target] += 1
            
    return class_counts

def plot_class_distribution(class_distribution, title):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_distribution)), class_distribution, align='center')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.show()