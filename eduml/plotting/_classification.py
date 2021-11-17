import numpy as np
from matplotlib import pyplot as plt

def plot_decision_regions(x, y, model, mesh_size=0.01, title='Unnamed'):
    redish = '#d73027'
    orangeish = '#fc8d59'
    yellowish = '#fee090'
    blueish = '#4575b4'

    colormap = np.array([blueish, redish])
    labels = np.array(['Class 1', 'Class 2'])

    x0 = x[:, 0]
    x1 = x[:, 1]
    y = np.array(y).astype(int)

    fig, ax = plt.subplots(figsize=(8,6), dpi=150)

    # meshgrid plotting for decision boundary
    PAD = 0.1 # relative padding
    x0_min, x0_max = x0.min()*(1-PAD), x0.max()*(1+PAD)
    x1_min, x1_max = x1.min()*(1-PAD), x1.max()*(1+PAD)
    x0_axis_range = np.arange(x0_min,x0_max, mesh_size)
    x1_axis_range = np.arange(x1_min,x1_max, mesh_size)

    xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)
    xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))

    pred_y = model.predict(xx).astype(int).reshape(-1,)

    if len(model.predict_proba(xx)) > 0:
        pred_s = model.predict_proba(xx) 
        ax.scatter(xx[:, 0], xx[:, 1], c=colormap[pred_y], s=50*pred_s**3, alpha=0.1, linewidths=0,)
    else:
        ax.scatter(xx[:, 0], xx[:, 1], c=colormap[pred_y], alpha=0.3, linewidths=0,)

    ax.contour(x0_axis_range, x1_axis_range,
               np.reshape(pred_y,(xx0.shape[0],-1)),
               levels=len(np.unique(y))-1, linewidths=1,
               colors=colormap[np.unique(pred_y)])

    for k, color, label in zip(np.unique(y), colormap, labels):
        ax.scatter(x0[y==k], x1[y==k], c=color, s=20, edgecolor='black', linewidth=1, label=label)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.legend(loc='best')

    return fig

def test():
    pass

if __name__ == '__main__':
    test()
