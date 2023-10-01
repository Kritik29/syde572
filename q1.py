import numpy as np
import matplotlib.pyplot as plt

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

# print(data_class1, "\n")
# print(data_class2, '\n')

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

# plotting the results for sample size = 5
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

print("Class 1: (mean, cov, eig_vals, eig_vecs) \n")
print(class1_hundred_samples_mean, '\n')
print(class1_hundred_samples_cov, '\n')
print(class1_eig_vals, '\n')
print(class1_eig_vecs, '\n')

print("Class 2: (mean, cov, eig_vals, eig_vecs) \n")
print(class2_hundred_samples_mean, '\n')
print(class2_hundred_samples_cov, '\n')
print(class2_eig_vals, '\n')
print(class2_eig_vecs, '\n')

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

plt.figure(5)
plt.scatter(data_class1[:,0], data_class1[:,1])
plt.scatter(data_class2[:,0], data_class2[:,1])
plt.legend(["Class 1", "Class 2"])
plt.plot(x, (1.12*x+59.55)/8.53, '-g')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('MED Classifier, Sample Size = 5')
plt.grid()
plt.show()
