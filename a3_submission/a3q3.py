import numpy as np

from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
###################################################################3

# Exercise 3

# set threshold for the negative-log likelihood value
tol = 1e-5
neg_log_likelihood_vals = []

np.random.seed(1)

# helper function to calculate mv gaussian
def multivariate_gaussian(x, mean, covariance):
  d = len(x)
  inv_covariance = 1 / covariance
  diff = x - mean
  exponent = -0.5 * np.sum(diff * (diff * inv_covariance))
  # using properties of diagonal covariance matrices to make computation a bit easier
  likelihood = (1 / np.sqrt((2 * np.pi) ** d * np.prod(covariance))) * np.exp(exponent)
  return likelihood

# EM algorithm for diagonal Gaussian Mixture Model

def EM_dGMM(X, K, max_iter=50):
  n, d = X.shape
  # random initialization of means, diagonal covariance matrices, and cluster coefficients
  means = X[np.random.choice(n, K, replace=False)]
  covariances = [np.diag(np.random.rand(d)) for _ in range(K)]
  cluster_coefficients = np.random.rand(K)

  likelihood = np.zeros((n, K))

  for iter in range(max_iter):
    # expectation step
    for k in range(K):
      # calculate responsibilities
      for i in range(n):
        likelihood[i, k] = multivariate_gaussian(X[i], means[k], covariances[k])
    responsibilities = cluster_coefficients * likelihood

    # maximization step
    N_k = responsibilities.sum(axis=0)
    # re-estimate the parameters and update the diagonal covariance matrices
    cluster_coefficients = N_k / n
    means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
    covariances = [np.dot(responsibilities[:, k] * (X - means[k]).T, (X - means[k])) / N_k[k] for k in range(K)]

    # compute negative-log likelihood, I used log1p to prevent divide-by-zero errors
    neg_log_likelihood = -1*np.log1p(np.dot(likelihood, cluster_coefficients)).sum()
    print(f"Iteration {iter + 1}: Negative-Log Likelihood = {neg_log_likelihood}")
    neg_log_likelihood_vals.append(neg_log_likelihood)

    # check negative-log likelihood for convergence
    if (iter > 1) & (abs((neg_log_likelihood_vals[iter])-(neg_log_likelihood_vals[iter-1]))<= tol*abs(neg_log_likelihood_vals[iter])):
      break
    
  return cluster_coefficients, means, covariances

# import the training and test data sets seperately
mnist_training_data = MNIST(root='./data', train=True, download=True, transform=None)
mnist_test_data = MNIST(root='./data', train=False, download=True, transform=None)

# store training data and labels
training_data = mnist_training_data.data.numpy()
training_labels = mnist_training_data.targets.numpy()

# store test data and labels
test_data = mnist_test_data.data.numpy()
test_labels = mnist_test_data.targets.numpy()

# reshape training and test data into 784x1 vector
training_data_flattened = training_data.reshape(-1, 28*28)
test_data_flattened = test_data.reshape(-1, 28*28)

K = 5 

# applying PCA on flattened data because I was getting overflow errors
pca = PCA(n_components=1)
training_data_pca = pca.fit_transform(training_data_flattened).astype(np.float32)
test_data_pca = pca.fit_transform(test_data_flattened).astype(np.float32)

# fit the model
cluster_coefficients, means, covariances = EM_dGMM(training_data_pca, K)

# bayes classifier for test image label prediction
def bayes_classifier(X):
  predicted_class = []
  for cluster in range(0, 9, 1):
    # calculating using the density of the cth cluster in the training set
    predicted_class.append((np.bincount(training_labels)[cluster]/np.bincount(training_labels).sum()) * X)
  # return the most likely class
  return np.argmax(predicted_class)

# applying the bayes classifier to each image in the test set, storing results in array
results = []
for image in test_data_pca:
  results.append(bayes_classifier(image))

# calculate and print the error
print(f"\nTest error, K = 5: {(1-(sum(results)/sum(test_labels))): .2%}" )
