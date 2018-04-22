
import numpy as np
import matplotlib.pyplot as plt


def train_test_splitter(X, y, ratio = 0.8, random_seed = 0):

    assert(len(X) == len(y)), "The number of points in feature matrix and target vector should be the same."
    np.random.seed(random_seed)
    
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)

    train_idx = idx[:int(n * ratio)]
    test_idx = idx[int(n * ratio):]

    return X[train_idx,:], X[test_idx,:], y[train_idx], y[test_idx]

def error_rate(y, y_predicted):
    
    assert len(y) == len(y_predicted), "The number of targets and predictions should be the same."
    assert len(y) != 0, "The number of targets and predictions should not be zero."
    
    return np.sum(np.array(y) != np.array(y_predicted)) / len(y)

def plot_losses(losses, savefig = False, showfig = False, filename = 'loss.png'):

    fig = plt.figure(figsize = (12,8))
    plt.plot(np.arange(len(losses)), losses, color = 'r', marker = 'o', label = 'Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of Iterations')

    if savefig:
        fig.savefig(filename, format = 'png', dpi = 600, bbox_inches = 'tight')
    if showfig:
        plt.show()
    plt.close()

    return 