import numpy as np
from matplotlib import pyplot as plt

def plot_decision_regions(x, y, model, mesh_size=0.1, title='Unnamed'):
    redish = '#d73027'
    orangeish = '#fc8d59'
    yellowish = '#fee090'
    blueish = '#4575b4'
    colormap = np.array([blueish, redish, yellowish, orangeish])

    x0 = x[:, 0]
    x1 = x[:, 1]
    y = np.array(y)

#plt.style.use('seaborn-whitegrid') # set style because it looks nice
    fig, ax = plt.subplots(figsize=(8,6), dpi=150)
    ax.scatter(x0, x1, c=colormap[y.astype(int)], edgecolor='black', s=10)

# meshgrid plotting for decision boundary
    PAD = 1 
    x0_min, x0_max = np.round(x0.min())-PAD, np.round(x0.max()+PAD)
    x1_min, x1_max = np.round(x1.min())-PAD, np.round(x1.max()+PAD)
    x0_axis_range = np.arange(x0_min,x0_max, mesh_size)
    x1_axis_range = np.arange(x1_min,x1_max, mesh_size)

    xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)
    xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))

    pred_y = model.predict(xx).astype(int).reshape(-1,)
    # pred_prob_y = model.predict_proba(xx)
    pred_s = model.predict_proba(xx) 

    ax.scatter(xx[:, 0], xx[:, 1], c=colormap[pred_y], s=20*pred_s**5, alpha=0.3, linewidths=0,)
    ax.contour(x0_axis_range, x1_axis_range,
               np.reshape(pred_y,(xx0.shape[0],-1)),
               levels=len(np.unique(y))-1, linewidths=1,
               colors=colormap[np.unique(pred_y)])

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')

    # plt.savefig('output_knn5.png')
    return fig
