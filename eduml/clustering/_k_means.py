import numpy as np 

class KMeans:
    def __init__(self, k=2):
       self.k = k 
       self.X = None
       self.clusters = None

       self.n = self.p = None

    def fit(self, X):
        self.X = X
        self.n, self.p = self.X.shape

        centroids = self.X[np.random.choice(self.n, size=self.k)]

        while True:
            distances = []
            for x in self.X:
                distance = []
                for centroid in centroids:
                    d = KMeans.euclidean_distance(centroid, x)
                    distance.append(d)
                distances.append(distance)

            distances = np.array(distances)
            self.clusters = np.argmin(distances, axis=1)

            prev_centroids = centroids
            centroids = []
            for k in range(self.k):
                centroids.append(np.mean(self.X[self.clusters==k], axis=0))
            centroids = np.array(centroids)

            if (prev_centroids == centroids).all():
                break

        return self.clusters

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((y-x)**2))




if __name__ == '__main__':
    from matplotlib import pyplot as plt

    colors = np.array(['red', 'blue'])
    mean0 = [2,2]
    mean1 = [6,6]
    cov = [[0.5, 0],[0,0.5]]
    class0 = np.random.multivariate_normal(mean0, cov, 100)
    class1 = np.random.multivariate_normal(mean1, cov, 100)

    y = np.hstack((np.ones(100), np.zeros(100))).astype(int)
    X = np.vstack((class0, class1))

    cluster = KMeans(k=2)
    clusters = cluster.fit(X)

    # print(np.vstack((y, clusters)).reshape(200, -1))
    correct = y == clusters
    c = sum(correct)

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].scatter(X[:, 0], X[:, 1], c=colors[y])
    ax[0].set_title('Two dimensional test data')
    ax[1].scatter(X[:,0], X[:,1], c=colors[clusters], edgecolor=np.array(['black', 'green']*200)[correct.astype(int)])
    ax[1].set_title(f'K-Means Clustering Result (Score={round(c/200*100)}%)')
    for a in ax:
        a.set_xlabel('X1')
        a.set_ylabel('X2')


    plt.show()
