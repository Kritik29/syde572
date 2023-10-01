import numpy as np
import matplotlib.pyplot as plt

# seeding the random number generator with a fixed value so the
# same random array is generated each time
np.random.seed(1)

samples = 5

mean_class1 = np.array([4, 7])
mean_class2 = np.array([5, 10])

covariance_class1 = np.array([[9,3],
                              [3,10]])

covariance_class2 = np.array([[7,0],
                              [0,16]])

data_class1 = np.random.multivariate_normal(mean_class1, covariance_class1, samples)
data_class2 = np.random.multivariate_normal(mean_class2, covariance_class2, samples)

x_class1, y_class1 = np.meshgrid(np.linspace(-1, 5, 500), np.linspace(-1, 7, 500))
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

def gaussian_pdf(x, mean, covariance):
    n = mean.shape[0]
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=2)
    return (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)

pdf_class1_hand_calc_5_samples = gaussian_pdf(pos_class1, mean_class1_hand_calc, covariance_class1_hand_calc)
pdf_class2_hand_calc_5_samples = gaussian_pdf(pos_class2, mean_class2_hand_calc, covariance_class2_hand_calc)

plt.figure(1)
contour_class1 = plt.contourf(x_class1, y_class1, pdf_class1_hand_calc_5_samples)
plt.colorbar(contour_class1, label='PDF Value')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Gaussian Random Data and Equiprobability Contours')
plt.grid()
plt.show()

plt.figure(2)
contour_class2 = plt.contourf(x_class2, y_class2, pdf_class2_hand_calc_5_samples)
plt.colorbar(contour_class2, label='PDF Value')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Gaussian Random Data and Equiprobability Contours')
plt.grid()
plt.show()



# print(data_class1, "\n")
# print(data_class2)