import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(1)

mean_class1 = np.array([4, 7])
mean_class2 = np.array([5, 10])

covariance_class1 = np.array([[9,3],
                              [3,10]])

covariance_class2 = np.array([[7,0],
                              [0,16]])

data_class1_100_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 100)
data_class2_100_samples = np.random.multivariate_normal(mean_class2, covariance_class2, 100)

training_data = np.vstack((data_class1_100_samples, data_class2_100_samples))
labels = np.array([1] * len(data_class1_100_samples)+ [2] * len(data_class2_100_samples))

x_class1, y_class1 = np.meshgrid(np.linspace(-3, 18, 500), np.linspace(-5, 25, 500))

class1_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)
class2_50_new_samples = np.random.multivariate_normal(mean_class2, covariance_class2, 50)

white_noise = np.random.normal(0, 1, size=(50, 2))

class1_50_new_samples_noisy = class1_50_new_samples + white_noise
class2_50_new_samples_noisy = class2_50_new_samples + white_noise

def knn(X_train, y_train, x, k):
    # Calculate the Euclidean distances between x and all training data points
    distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
    
    # Sort distances and get the indices of the k-nearest neighbors
    k_indices = np.argsort(distances)[:k]
    
    # Get the labels of the k-nearest neighbors
    k_nearest_labels = y_train[k_indices]
    
    # Count the occurrences of each label
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    # Find the label with the most occurrences among the k-nearest neighbors
    most_common_label = unique_labels[np.argmax(counts)]
    return most_common_label

# print(np.vstack((data_class1_100_samples, data_class2_100_samples)))
# print('\n')
# print(labels)

grid_points = np.c_[x_class1.ravel(), y_class1.ravel()]
Z = np.array([knn(training_data, labels, point, 1) for point in grid_points])
Z = Z.reshape(x_class1.shape)
cmap = ListedColormap(['#1b5194', '#f29249'])

plt.figure(1)
plt.contourf(x_class1, y_class1, Z, cmap=cmap, alpha=0.4)
# plt.scatter(training_data[:,0], training_data[:,1], c=labels, cmap=cmap, edgecolor='k')
plt.scatter(data_class1_100_samples[:,0], data_class1_100_samples[:,1])
plt.scatter(data_class2_100_samples[:,0], data_class2_100_samples[:,1])
# plt.scatter(class1_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.scatter(class2_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.scatter(class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(["Class 1 Data", "Class 2 Data"])
plt.title('kNN Classifier Boundaries, k = 1')
plt.grid()
plt.show()