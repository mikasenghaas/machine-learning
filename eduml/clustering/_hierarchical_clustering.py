import numpy as np 

class BottomUpHierachicalClustering:
    """
    Implementation of the bottom up hierarchical clustering algorithm.
    Used in unsupervised learning setting to detect any number k clusters within 
    p-dimensional data. The clusters are initialised as singletons in the number of 
    data points and then continuously merged, until there either exists a single 
    cluster or the minimal cluster number min-clusters (in .fit()) was reached.

    For a single merge from k -> k-1 clusters, the distance between all unique pairs of
    clusters is measured and the most similar cluster pair is merged:
    The cluster distance is a hyper parameter of the algorithm and can be set:

    self.cluster_algorithm : str
    Determines how to compute the distance between two clusters C1 and C2

    'single-link': min(dist(x,y) | x in C1, y in C2)
    'complete-link': max(dist(x,y) | x in C1, y in C2)
    'group-average': mean(dist(x,y) | x in C1, y in C2)

    The single-link algorithm tends to find highly unbalanced classes and detect outliers,
    while the other two cluster measurements tend to create equally large clusters.

    self.clusters : dict
    Stores all cluster mapping (array of length n, indicating which cluster the
    data point at index i belongs to) for each number of clusters.
    key : number of clusters k 
    value : array of cluster mapping 
    """

    def __init__(self, cluster_distance='group_average'):
        self.X = None
        self.n = None
        self.p = None

        # hierarchical clustering specific attributes 
        self.cluster_distance = cluster_distance 
        self.clusters = None # dict of num of clusters


    def fit(self, X, min_clusters=1):
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = X
        self.n, self.p = X.shape

        self.clusters = {i: (None if i != self.n else np.array(list(range(self.n))))
                            for i in range(1, self.n+1)}

        dist = []
        for x in self.X:
            line = []
            for y in self.X:
                line.append(HierachicalClustering.euclidean_distance(x, y))
            dist.append(line)
        dist = np.array(dist)

        for i in tqdm(range(self.n-1, min_clusters-1, -1)):
            cluster_distances = {} 
            # iterate over all pairs of clusters (number of i from previous iter)
            for c1 in range(i+1):
                for c2 in range(c1, i+1):
                    if c1 != c2:
                        # get indices for datapoints in curr pair of clusters
                        c1_indices = [j for j in range(len(self.clusters[i+1]))
                                        if self.clusters[i+1][j]==c1] 
                        c2_indices = [j for j in range(len(self.clusters[i+1]))
                                        if self.clusters[i+1][j]==c2] 
                        
                        # compute the list of distances between each pair of point in 
                        # the clusters 
                        cluster_dist = [HierachicalClustering.euclidean_distance(
                                            self.X[c1_ind], self.X[c2_ind]) 
                                            for c1_ind in c1_indices for c2_ind in c2_indices]
                        if self.cluster_distance == 'single-link':
                            dist = min(cluster_dist)
                        elif self.cluster_distance == 'complete-link':
                            dist = max(cluster_dist)
                        elif self.cluster_distance == 'group-average':
                            dist = sum(cluster_dist) / len(cluster_dist)

                        cluster_distances[(c1, c2)] = dist

            # find clusters that are closest
            #print(min(cluster_distances.values()))
            # print(f'test: {HierachicalClustering.euclidean_distance(self.X[2], self.X[0])}')
            merge = min(cluster_distances, key=cluster_distances.get)
            min_merge = min(merge)
            max_merge = max(merge)
            new_clusters = []
            for val in self.clusters[i+1]:
                if val == max_merge:
                    new_clusters.append(min_merge)
                elif val > max_merge:
                    new_clusters.append(val-1)
                else:
                    new_clusters.append(val)
            self.clusters[i] = new_clusters 

    def get_clusters(self, k):
        return np.array(self.clusters[k])

    @staticmethod
    def euclidean_distance(x, y):
        x, y = np.array(x), np.array(y)
        return np.sqrt(np.sum((y-x)**2))


if __name__ == '__main__':
    from sklearn import datasets
    from matplotlib import pyplot as plt

    SHOW_FIGURES = True

    colors = np.array(['red', 'blue'])
    n = 50
    mean0 = [2,2]
    mean1 = [4,4]
    cov = [[0.5, 0],[0,0.5]]
    class0 = np.random.multivariate_normal(mean0, cov,n )
    class1 = np.random.multivariate_normal(mean1, cov, n)

    X = np.vstack((class0, class1))
    y = np.hstack((np.zeros(n), np.ones(n))).astype(int)

    clu = ButtomUpHierachicalClustering(cluster_distance='group-average')
    clu.fit(X)
    clusters = clu.get_clusters(2)

    correct = y == clusters
    c = np.sum(correct)

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].scatter(X[:, 0], X[:, 1], c=colors[y])
    ax[0].set_title('Two dimensional test data')
    ax[1].scatter(X[:,0], X[:,1], c=colors[clusters], 
            edgecolor=np.array(['black', 'green']*200)[correct.astype(int)])
    ax[1].set_title(f'Hierarchical Clustering Result (Score={round(c/(n*2)*100)}%)')
    for a in ax:
        a.set_xlabel('X1')
        a.set_ylabel('X2')

    if SHOW_FIGURES:
        plt.show()
