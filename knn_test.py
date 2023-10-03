import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

mean_class1 = np.array([4, 7])
mean_class2 = np.array([5, 10])

covariance_class1 = np.array([[9,3],
                              [3,10]])

covariance_class2 = np.array([[7,0],
                              [0,16]])

data_class1_100_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 100)
data_class2_100_samples = np.random.multivariate_normal(mean_class2, covariance_class2, 100)

data_class2_100_samples_labelled = np.c_[data_class2_100_samples, np.zeros(100)]

x_class1, y_class1 = np.meshgrid(np.linspace(-1, 10, 500), np.linspace(-1, 12, 500))

class KNearestNeighbour:
  
  def __init__(self, n_neighbours=5, p=2):
    self.n_neighbours = n_neighbours
    self.p = p

  def fit(self, X, y):
    self.X = X
    self.y = y
    return self
  
  def predict(self, X):
    predictions = []
    for pred_row in X:
      euclidean_distances = []
      for X_row in self.X:
        distance = np.linalg.norm(X_row - pred_row, ord=self.p)
        euclidean_distances.append(distance)

      neighbours = self.y[np.argsort(euclidean_distances)[:self.n_neighbours]]
      neighbours_bc = np.bincount(neighbours)
      prediction = np.argmax(neighbours_bc)
      predictions.append(prediction)
    
    return predictions

knn = KNearestNeighbour()
knn.fit(data_class1_100_samples, data_class2_100_samples_labelled)
y_pred = knn.predict(x_class1)

# plt.figure(1)
# plt.scatter(data_class1_100_samples[:,0],data_class1_100_samples[:,1])
# plt.scatter(data_class2_100_samples[:,0], data_class2_100_samples[:,1])
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.grid()
# plt.show()

