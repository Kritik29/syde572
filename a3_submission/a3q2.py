import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
#############################################################################################

# Exercise 2

# import the training and test data sets seperately
mnist_training_data = MNIST(root='./data', train=True, download=True, transform=None)
mnist_test_data = MNIST(root='./data', train=False, download=True, transform=None)

# store training data and labels
training_data = mnist_training_data.data.numpy()
training_labels = mnist_training_data.targets.numpy()

# reshape training data into 784x1 vector
training_data_flattened = training_data.reshape(-1, 28*28)

# initialize array of k, which is the number of clusters
K = [5,10,20,40]

# perform k-means clustering for each k, and calculate cluster consistency for each k
for x in range(0, 4, 1):
  # set limit to how many iterations, saves time in testing. 
  # assumption is that the centroids converge in at most 10 iterations
  max_iterations = 10

  # pick centroids randomly since we are not told explicitly where to place them
  np.random.seed(1)
  centroids = training_data_flattened[np.random.choice(training_data_flattened.shape[0], K[x], replace=False)]

  # convergence tolerance
  tolerance = 1e-4

  # k-means clustering, runs only for the predefined number of iterations
  for j in range(max_iterations):
    # initialize array for distances
    distances = np.zeros((training_data_flattened.shape[0], K[x]))
    for i in range(K[x]):
      # update the distances from data points to cluster centroids
      distances[:, i] = np.linalg.norm(training_data_flattened - centroids[i], axis=1)

    # update labels to closest centroid
    labels = np.argmin(distances, axis=1)
    # update centroids
    new_centroids = np.array([training_data_flattened[labels == k].mean(axis=0) for k in range(K[x])])
    # used to check for convergence
    delta_centroids = new_centroids - centroids
    
    # check if centroids are changing (comparing with a tolerance value that is very small)
    if delta_centroids.all() < tolerance:
      break

    # update centroids
    centroids = new_centroids

  # use PCA for plotting purposes only, 1-d data is easily visualized
  pca = PCA(n_components=1)
  training_data_pca = pca.fit_transform(training_data_flattened)
  plt.plot(training_data_pca, labels, 'bo')
  plt.scatter(pca.transform(centroids)[:, 0], range(K[x]), c='red', marker='x', s=200, label='Centroids')
  plt.legend()
  plt.title(f'K-means Clustering, k = {K[x]}')
  plt.show()

  # initialize array to store consistency for each cluster
  cluster_consistency = []

  for cluster in range(K[x]):
    # associate data points to clusters
    cluster_indices = np.where(labels == cluster)[0]
    # determine number of points in cluster
    cluster_size = len(cluster_indices)
    # skip empty clusters
    if cluster_size == 0:
      continue

    # determine most occuring class in cluster
    class_counts = np.bincount(training_labels[cluster_indices])
    most_common_class = np.argmax(class_counts)

    # calculate consistency for each cluster and append to array
    consistency = class_counts[most_common_class] / cluster_size
    cluster_consistency.append(consistency)

  # report overall consistency for each k value
  overall_consistency = np.mean(cluster_consistency)
  print(f"Overall Cluster Consistency, k = {K[x]}: {overall_consistency: .1%}")
