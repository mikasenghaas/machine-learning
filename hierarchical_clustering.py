import numpy as np 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from icecream import ic
from tqdm import tqdm

class HierachicalClustering:
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

def main():
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

    clu = HierachicalClustering(cluster_distance='group-average')
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

if __name__ == '__main__':
    main()
