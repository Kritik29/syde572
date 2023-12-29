import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
####################################################3

# Exercise 1

np.random.seed(1)

samples = 5

mean_class1 = np.array([1, 3])
mean_class2 = np.array([20, 31])

covariance_class1 = np.array([[1,0],
                              [0,15]])

covariance_class2 = np.array([[3,4],
                              [4,11]])

mean_noise = np.array([2,2])
covariance_noise = np.array([[2,0], 
                             [0,3]])

# Sample size = 5 
data_class1 = np.random.multivariate_normal(mean_class1, covariance_class1, samples)
data_class2 = np.random.multivariate_normal(mean_class2, covariance_class2, samples)

noise_class1 = np.random.multivariate_normal(mean_noise, covariance_noise, 5)
noise_class2 = np.random.multivariate_normal(mean_noise, covariance_noise, 5)

data_class1_w_noise = data_class1 + noise_class1
data_class2_w_noise = data_class2 + noise_class2

x = np.linspace(-5, 30, 1000)
x_100_samples = np.linspace(-3, 12, 1000)

mmd = -((-85*x)-(13622)+(((52153*(x**2))+(778684*x)+12050404)**0.5))/432

plt.figure(1)
plt.scatter(data_class1_w_noise[:,0], data_class1_w_noise[:,1])
plt.scatter(data_class2_w_noise[:,0], data_class2_w_noise[:,1])
plt.plot(x, mmd, '-g')
plt.legend(["Class 1", "Class 2"])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('MMD Classifier, Sample Size = 5')
plt.grid()
# plt.show()

data_class1_100 = np.random.multivariate_normal(mean_class1, covariance_class1, 100)
data_class2_100 = np.random.multivariate_normal(mean_class2, covariance_class2, 100)

noise_class1_100 = np.random.multivariate_normal(mean_noise, covariance_noise, 100)
noise_class2_100 = np.random.multivariate_normal(mean_noise, covariance_noise, 100)

data_class1_w_noise_100 = data_class1_100 + noise_class1_100
data_class2_w_noise_100 = data_class2_100 + noise_class2_100

class1_hundred_samples_mean = np.mean(data_class1_w_noise_100, axis=0)
class2_hundred_samples_mean = np.mean(data_class2_w_noise_100, axis=0)

# determining sample covariance
class1_hundred_samples_cov = np.cov(data_class1_w_noise_100[:,0], data_class1_w_noise_100[:,1])
class2_hundred_samples_cov = np.cov(data_class2_w_noise_100[:,0], data_class2_w_noise_100[:,1])

# define symbolic variables
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')

X = np.array([x1, x2])

mmd_boundary = np.matmul(X.T, (np.linalg.inv(class1_hundred_samples_cov))-(np.linalg.inv(class2_hundred_samples_cov)), X)+ 2*np.matmul(class2_hundred_samples_mean.T, (np.linalg.inv(class2_hundred_samples_cov)))*X - 2*np.matmul(class1_hundred_samples_mean.T, (np.linalg.inv(class1_hundred_samples_cov)))*X +np.matmul(class1_hundred_samples_mean, (np.linalg.inv(class1_hundred_samples_cov)), class1_hundred_samples_mean) - np.matmul(class2_hundred_samples_mean, (np.linalg.inv(class2_hundred_samples_cov)), class2_hundred_samples_mean)

mmd_boundary_100 = sym.solve(mmd_boundary, x1)

plt.figure(2)
plt.scatter(data_class1_w_noise_100[:,0], data_class1_w_noise_100[:,1])
plt.scatter(data_class2_w_noise_100[:,0], data_class2_w_noise_100[:,1])
plt.plot(x, -((-85*x)-(13600)+(((52148*(x**2))+(778683*x)+12050432)**0.5))/444, '-g')
plt.legend(["Class 1", "Class 2"])
plt.title("MMD Classifier, 100 samples")
plt.grid()
# plt.show()

class1_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)
class2_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)

# generate white noise
white_noise = np.random.multivariate_normal([0,0], [[1,0], [0,1]])

# add noise to the 50 new samples
class1_50_new_samples_noisy = class1_50_new_samples + white_noise
class2_50_new_samples_noisy = class2_50_new_samples + white_noise

plt.figure(3)
plt.scatter(class1_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
plt.scatter(class2_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
plt.plot(x, mmd, '-g')
plt.plot(x_100_samples, -((-85*x)-(13600)+(((52148*(x**2))+(778683*x)+12050432)**0.5))/444, '-r')
plt.legend(["Class 1 Noise Samples", "Class 2 Noise Samples", "n = 5 decision boundary", "n = 100 decision boundary"])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Noisy Samples')
plt.grid()
# plt.show()

########################################################################33

# Exercise 2

# ML classifier based on normal distribution
def MLClassifier(x, mean, cov):
    d = x - mean
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * np.dot(np.dot(d, inv_cov), d)
    return np.exp(exponent) / np.sqrt((2 * np.pi) ** len(mean) * np.linalg.det(cov))

MLClassificationResults_class1 = []
MLClassificationResults_class2 = []

for new_data in class1_50_new_samples_noisy:
    likelihood_class1 = MLClassifier(new_data, mean_class1, covariance_class1)
    likelihood_class2 = MLClassifier(new_data, mean_class2, covariance_class2)

    if likelihood_class1 > likelihood_class2:
        MLClassificationResults_class1.append(1)
    else:
        MLClassificationResults_class1.append(2)

for new_data in class2_50_new_samples_noisy:
    likelihood_class1_2 = MLClassifier(new_data, mean_class1, covariance_class1)
    likelihood_class2_2 = MLClassifier(new_data, mean_class2, covariance_class2)

    if likelihood_class1_2 > likelihood_class2_2:
        MLClassificationResults_class2.append(1)
    else:
        MLClassificationResults_class2.append(2)

# print(MLClassificationResults_class1)
# print("\n")
# print(MLClassificationResults_class2)

def MAPClassifier(new_point, class_means, class_covariances, priors):
    num_classes = len(class_means)
    posteriors = []

    for i in range(num_classes):
        mean = class_means[i]
        covariance = class_covariances[i]
        prior_prob = priors[i]
        diff = new_point - mean
        likelihood = (1 / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))) * np.exp(-0.5 * np.dot(np.dot(diff, np.linalg.inv(covariance)), diff.T))

        # Calculate posterior probability
        posterior = likelihood * prior_prob
        posteriors.append(posterior)

    # Return the class with the highest posterior probability
    predicted_class = np.argmax(posteriors)
    # Adding +1 to the return because argmax will return either 0 or 1 (based on index), but we have
    # class 1 or class 2. So to be consistent we need to add 1
    return predicted_class+1

for x in class1_50_new_samples_noisy:
    predicted_class = MAPClassifier(x, [mean_class1, mean_class2], [covariance_class1, covariance_class2], [0.58, 0.42])
    # print(predicted_class)

print("\n")

for x in class2_50_new_samples_noisy:
    predicted_class = MAPClassifier(x, [mean_class1, mean_class2], [covariance_class1, covariance_class2], [0.58, 0.42])
    # print(predicted_class)

#############################################################################333

# Exercise 3

# Generating 100 samples based on the statistics given
q3_class_1_data = np.random.normal(0.5, 1, 100)
q3_class_2_data = np.random.normal(5, 3, 100)

# ML classifier based on exponential distribution
def MLClassifier_exp_dist(class_1_data, class_2_data, x):
    lambda_class_1 = 1 / np.mean(class_1_data)
    lambda_class_2 = 1 / np.mean(class_2_data)

    # Likelihood functions for each class
    likelihood_class_1 = lambda_class_1 * np.exp(-lambda_class_1 * x)
    likelihood_class_2 = lambda_class_2 * np.exp(-lambda_class_2 * x)

    if likelihood_class_1 > likelihood_class_2:
        return 1
    else:
        return 2
    
# ML classifier based on uniform distribution
def MLClassifier_uniform_dist(class_1_data, class_2_data, x):
    min_class_1 = min(class_1_data)
    max_class_1 = max(class_1_data)
    min_class_2 = min(class_2_data)
    max_class_2 = max(class_2_data)

    likelihood_class_1 = 1 / (max_class_1 - min_class_1)
    likelihood_class_2 = 1 / (max_class_2 - min_class_2)

    if min_class_1 <= x <= max_class_1:
        likelihood_class_1 /= (max_class_1 - min_class_1)

    if min_class_2 <= x <= max_class_2:
        likelihood_class_2 /= (max_class_2 - min_class_2)

    if likelihood_class_1 > likelihood_class_2:
        return 1
    else:
        return 2

# ML classifier based on combined Gaussian and Exponential distribution
def MLClassifier_combined_dist(class_1_data, class_2_data, x):
    mean_class_1 = np.mean(class_1_data)
    variance_class_1 = np.var(class_1_data)
    lambda_class_1 = 1 / mean_class_1
    
    mean_class_2 = np.mean(class_2_data)
    variance_class_2 = np.var(class_2_data)
    lambda_class_2 = 1 / mean_class_2

    # definitions of likelihood functions 
    likelihood_class_1 = 0.5*(1 / np.sqrt(2 * np.pi * variance_class_1))*lambda_class_1 * np.exp(-0.5 * ((x - mean_class_1) ** 2) / variance_class_1)*lambda_class_1 * np.exp(-lambda_class_1 * x)
    likelihood_class_2 = 0.5*(1 / np.sqrt(2 * np.pi * variance_class_2))*lambda_class_2 * np.exp(-0.5 * ((x - mean_class_2) ** 2) / variance_class_2)*lambda_class_2 * np.exp(-lambda_class_2 * x)

    if likelihood_class_1 > likelihood_class_2:
        return 1
    else:
        return 2

# Defining empty arrays to hold the classification values
MLClassifier_exp_results_class1 = []
MLClassifier_exp_results_class2 = []

MLClassifier_uni_results_class1 = []
MLClassifier_uni_results_class2 = []

MLClassifier_combined_results_class1 = []
MLClassifier_combined_results_class2 = []

# Generate white noise
q3_white_noise = np.random.normal(0, 1, 50)

# Noisy data
q3_class_1_test_data = np.random.normal(0.5, 1, 50) + q3_white_noise
q3_class_2_test_data = np.random.normal(5, 3, 50) + q3_white_noise

# Run each type of classifier over both sets of test data
for x in q3_class_1_test_data:
    MLClassifier_exp_results_class1.append(MLClassifier_exp_dist(q3_class_1_data, q3_class_2_data, x))

for x in q3_class_2_test_data:
    MLClassifier_exp_results_class2.append(MLClassifier_exp_dist(q3_class_1_data, q3_class_2_data, x))

for x in q3_class_1_test_data:
    MLClassifier_uni_results_class1.append(MLClassifier_uniform_dist(q3_class_1_data, q3_class_2_data, x))

for x in q3_class_2_test_data:
    MLClassifier_uni_results_class2.append(MLClassifier_uniform_dist(q3_class_1_data, q3_class_2_data, x))

for x in q3_class_1_test_data:
    MLClassifier_combined_results_class1.append(MLClassifier_combined_dist(q3_class_1_data, q3_class_2_data, x))

for x in q3_class_2_test_data:
    MLClassifier_combined_results_class2.append(MLClassifier_combined_dist(q3_class_1_data, q3_class_2_data, x))
  
print("ML Classifier Exp. Dist. Accuracy, class 1: ", MLClassifier_exp_results_class1.count(1)/50, "\n")
print("ML Classifier Exp. Dist. Accuracy, class 2: ", MLClassifier_exp_results_class2.count(2)/50, "\n")
print("ML Classifier Uni. Dist. Accuracy, class 1: ", MLClassifier_uni_results_class1.count(1)/50, "\n")
print("ML Classifier Uni. Dist. Accuracy, class 2: ", MLClassifier_uni_results_class2.count(2)/50, "\n")
print("ML Classifier Combined Dist. Accuracy, class 1: ", MLClassifier_combined_results_class1.count(1)/50, "\n")
print("ML Classifier Combined Dist. Accuracy, class 2: ", MLClassifier_combined_results_class2.count(2)/50, "\n")