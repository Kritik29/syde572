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

print(data_class1, "\n")
print(data_class2)