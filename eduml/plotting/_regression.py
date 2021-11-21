from matplotlib import pyplot as plt
import numpy as np 

def plot_1d_regression(X, y, model, figsize=(8,6), title='unnamed'):
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(X, y, c='b', label='$X_1$');
    x_range = np.linspace(min(X), max(X), 1000)
    ax.plot(x_range, model.predict(x_range), c='red', label='$\hat{y}$');
    ax.set_title(title);
    ax.set_xlabel('$X_1$');
    ax.set_ylabel('$y$');
    ax.grid()

    return fig
