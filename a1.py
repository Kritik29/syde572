import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from matplotlib.colors import ListedColormap

############################################################################

# Exercise 1

# seeding the random number generator with a fixed value so the
# same random array is generated each time
np.random.seed(1)

samples = 5

# defining the class mean and covariances as per the definition
mean_class1 = np.array([4, 7])
mean_class2 = np.array([5, 10])

covariance_class1 = np.array([[9,3],
                              [3,10]])

covariance_class2 = np.array([[7,0],
                              [0,16]])

# function to determine the Gaussian PDF values
def gaussian_pdf(x, mean, covariance):
    n = mean.shape[0]
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=2)
    return (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)

# Sample size = 5 
data_class1 = np.random.multivariate_normal(mean_class1, covariance_class1, samples)
data_class2 = np.random.multivariate_normal(mean_class2, covariance_class2, samples)

# defining the plot
x_class1, y_class1 = np.meshgrid(np.linspace(-1, 10, 500), np.linspace(-1, 12, 500))
pos_class1 = np.dstack((x_class1, y_class1))

x_class2, y_class2 = np.meshgrid(np.linspace(0, 8, 500), np.linspace(4, 16, 500))
pos_class2 = np.dstack((x_class2, y_class2))

# mean and covariances based off the 5 samples, calculated by hand
mean_class1_hand_calc = np.array([4.09, 3.18])
mean_class2_hand_calc = np.array([2.97, 11.71])

covariance_class1_hand_calc = np.array([[5.99, 2.93],
                                        [2.93, 6.22]])

covariance_class2_hand_calc = np.array([[5.18, -4.9], 
                                        [-4.9, 8.56]])

# determining the gaussian pdf values
pdf_class1_hand_calc_5_samples = gaussian_pdf(pos_class1, mean_class1_hand_calc, covariance_class1_hand_calc)
pdf_class2_hand_calc_5_samples = gaussian_pdf(pos_class2, mean_class2_hand_calc, covariance_class2_hand_calc)

# plotting the equiprobability contours for sample size = 5, for both class 1 and class 2

# plt.figure(1)
# contour_class1 = plt.contourf(x_class1, y_class1, pdf_class1_hand_calc_5_samples)
# plt.colorbar(contour_class1, label='PDF Value')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Gaussian Random Data and Equiprobability Contours')
# plt.grid()
# plt.show()

# plt.figure(2)
# contour_class2 = plt.contourf(x_class2, y_class2, pdf_class2_hand_calc_5_samples)
# plt.colorbar(contour_class2, label='PDF Value')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Gaussian Random Data and Equiprobability Contours')
# plt.grid()
# plt.show()

# Sample size = 100

data_class1_100_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 100)
data_class2_100_samples = np.random.multivariate_normal(mean_class2, covariance_class2, 100)

# determining sample mean
class1_hundred_samples_mean = np.mean(data_class1_100_samples, axis=0)
class2_hundred_samples_mean = np.mean(data_class2_100_samples, axis=0)

# determining sample covariance
class1_hundred_samples_cov = np.cov(data_class1_100_samples[:,0], data_class1_100_samples[:,1])
class2_hundred_samples_cov = np.cov(data_class2_100_samples[:,0], data_class2_100_samples[:,1])

# determining the gaussian pdf values
pdf_class1_hundred_samples = gaussian_pdf(pos_class1, class1_hundred_samples_mean, class1_hundred_samples_cov)
pdf_class2_hundred_samples = gaussian_pdf(pos_class2, class2_hundred_samples_mean, class1_hundred_samples_cov)

# determining the eigvals and eigvecs for each class
class1_eig_vals, class1_eig_vecs = np.linalg.eig(class1_hundred_samples_cov)
class2_eig_vals, class2_eig_vecs = np.linalg.eig(class2_hundred_samples_cov)

# print("Class 1: (mean, cov, eig_vals, eig_vecs) \n")
# print(class1_hundred_samples_mean, '\n')
# print(class1_hundred_samples_cov, '\n')
# print(class1_eig_vals, '\n')
# print(class1_eig_vecs, '\n')

# print("Class 2: (mean, cov, eig_vals, eig_vecs) \n")
# print(class2_hundred_samples_mean, '\n')
# print(class2_hundred_samples_cov, '\n')
# print(class2_eig_vals, '\n')
# print(class2_eig_vecs, '\n')

# plot the equiprobability contours for sample size = 100, for both class 1 and 2

# plt.figure(3)
# contour_class1_hundred_samples = plt.contourf(x_class1, y_class1, pdf_class1_hundred_samples)
# # plt.scatter(data_class1_100_samples[:,0], data_class1_100_samples[:,1])
# plt.colorbar(contour_class1_hundred_samples, label='PDF Value')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Gaussian Random Data and Equiprobability Contours')
# plt.grid()
# plt.show()

# plt.figure(4)
# contour_class2_hundred_samples = plt.contourf(x_class2, y_class2, pdf_class2_hundred_samples)
# # plt.scatter(data_class2_100_samples[:,0], data_class2_100_samples[:,1])
# plt.colorbar(contour_class2_hundred_samples, label='PDF Value')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Gaussian Random Data and Equiprobability Contours')
# plt.grid()
# plt.show()

##################################################################################################

# Exercise 2

x = np.linspace(-1, 8, 1000)
x_100_samples = np.linspace(-3, 12, 1000)

# plot MED classifier with 5 samples

# plt.figure(5)
# plt.scatter(data_class1[:,0], data_class1[:,1])
# plt.scatter(data_class2[:,0], data_class2[:,1])
# plt.legend(["Class 1", "Class 2"])
# plt.plot(x, (1.12*x+59.55)/8.53, '-g')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('MED Classifier, Sample Size = 5')
# plt.grid()
# plt.show()

# define symbolic variables
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')

X = np.array([x1, x2])

# define g1 and g2 and use the sample means to find the functions
g1 = -1*np.matmul(class1_hundred_samples_mean.T, X) + 0.5*np.matmul(class1_hundred_samples_mean.T, class1_hundred_samples_mean)
g2 = -1*np.matmul(class2_hundred_samples_mean.T, X) + 0.5*np.matmul(class2_hundred_samples_mean.T, class2_hundred_samples_mean)

# solve the equation g = g1-g2 to get the MED boundary condition
g = sym.solve(g1-g2, x2)

print("MED decision boundary, 100 samples: \n", g)

# plot MED classifier with 100 samples

# plt.figure(6)
# plt.scatter(data_class1_100_samples[:,0], data_class1_100_samples[:,1])
# plt.scatter(data_class2_100_samples[:,0], data_class2_100_samples[:,1])
# plt.legend(["Class 1", "Class 2"])
# plt.plot(x_100_samples, (10.702-(0.569*x_100_samples)), '-g')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('MED Classifier, Sample Size = 100')
# plt.grid()
# plt.show()

class1_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)
class2_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)

# generate white noise
white_noise = np.random.normal(0, 1, size=(50, 2))

# add noise to the 50 new samples
class1_50_new_samples_noisy = class1_50_new_samples + white_noise
class2_50_new_samples_noisy = class2_50_new_samples + white_noise

# plot both MED classification boundaries on the same plot

# plt.figure(7)
# plt.scatter(class1_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.scatter(class2_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.plot(x, (1.12*x+59.55)/8.53, '-g')
# plt.plot(x_100_samples, (10.702-(0.569*x_100_samples)), '-r')
# plt.legend(["Class 1 Noise Samples", "Class 2 Noise Samples", "n = 5 decision boundary", "n = 100 decision boundary"])
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Noisy Samples')
# plt.grid()
# plt.show()

##############################################################################################################

# Exercise 3

# define the meshgrid for this question, we'll use later when plotting
x_class1_exer3, y_class1_exer3 = np.meshgrid(np.linspace(-5, 18, 500), np.linspace(-5, 25, 500))

def knn(training_data, labels, x, k):
    # calculate distances between the point to classify and all training points
    # then, sort the distances and get the indices of the k-nearest neighbours
    distances = np.sqrt(np.sum((training_data - x) ** 2, axis=1))
    k_indices = np.argsort(distances)[:k]
    
    # find the labels of the k-nearest neighbours by looking into the labels array
    labels_of_k_nearest_neighbours = labels[k_indices]
    
    # count the occurrences of each label, find the one with the most occurences
    # and return the label
    unique_labels, counts = np.unique(labels_of_k_nearest_neighbours, return_counts=True)
    most_common_label = unique_labels[np.argmax(counts)]
    return most_common_label

# combine the class 1 and class 2 data into one array to feed into the kNN classifier
training_data = np.vstack((data_class1_100_samples, data_class2_100_samples))
# create an array of the labels, which will be used in the kNN classifier
labels = np.array([1] * len(data_class1_100_samples)+ [2] * len(data_class2_100_samples))

# create a grid of points for which the kNN classifier's boundary
grid_points = np.c_[x_class1_exer3.ravel(), y_class1_exer3.ravel()]
# for each point in the grid, determine the label based on the training dataset
# knn_boundary array is reshaped so it can be plotted
knn_boundary = np.array([knn(training_data, labels, point, 3) for point in grid_points])
knn_boundary = knn_boundary.reshape(x_class1.shape)
# colour map of size 2 to visualize the two classes
cmap = ListedColormap(['#1b5194', '#f29249'])

# plot kNN boundaries
plt.figure(8)
plt.contourf(x_class1_exer3, y_class1_exer3, knn_boundary, cmap=cmap, alpha=0.4)
# plt.scatter(data_class1_100_samples[:,0], data_class1_100_samples[:,1])
# plt.scatter(data_class2_100_samples[:,0], data_class2_100_samples[:,1])
plt.scatter(class1_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
plt.scatter(class2_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(["Class 1 Noisy Data", "Class 2 Noisy Data"])
plt.title('kNN Classifier Boundaries, k = 3')
plt.grid()
plt.show()
