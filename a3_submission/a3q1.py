import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA
from torchvision.datasets import MNIST

#############################################################################################

# Exercise 1

# import the training and test data sets seperately
mnist_training_data = MNIST(root='./data', train=True, download=True, transform=None)
mnist_test_data = MNIST(root='./data', train=False, download=True, transform=None)

# since we are instructed to only use the first two classes, we will only select images and labels
# from those classes

# to get the probability distribution for each class, I generated the plots two times.
class_0 = (0)
subset_idx = torch.isin(mnist_training_data.targets, torch.as_tensor(class_0))
train_images_class_0 = mnist_training_data.data[subset_idx].numpy()
train_labels_class_0 = mnist_training_data.targets[subset_idx].numpy()

class_1 = (1)
subset_id_1 = torch.isin(mnist_training_data.targets, torch.as_tensor(class_1))
train_images_class_1 = mnist_training_data.data[subset_id_1].numpy()
train_labels_class_1 = mnist_training_data.targets[subset_id_1].numpy()

# flatten the images into 784x1 vectors
train_images_class_0_flattened = train_images_class_0.reshape(-1, 28*28)
train_images_class_1_flattened = train_images_class_1.reshape(-1, 28*28)

# use PCA to convert 784x1 vectors into 1x1 vectors (announcement made on LEARN about this)
pca = PCA(n_components=1)
train_images_class_0_pca = pca.fit_transform(train_images_class_0_flattened).astype(np.float32)
train_images_class_1_pca = pca.fit_transform(train_images_class_1_flattened).astype(np.float32)

# store all region sizes
region_sizes = [1, 10, 100]

# find probability distribution for each region size
# I repeat this process for both the classes, hence the two loops
for region_size in region_sizes:
  # number of bins will be at least 1
  number_of_bins = max(int((max(train_images_class_0_pca) - min(train_images_class_0_pca)) / region_size), 1)
  histogram, bin_edges = np.histogram(train_images_class_0_pca, bins=number_of_bins, density=True)

  # plot the probability distribution for each region size
  plt.subplot(len(region_sizes), 1, region_sizes.index(region_size) + 1)
  plt.bar(bin_edges[:-1], histogram, width=region_size, align='center', alpha=0.5, color='b')
  plt.title(f'Class 0, Region Size: {region_size}')

plt.tight_layout()
plt.show()

for region_size in region_sizes:
  # number of bins will be at least 1
  number_of_bins = max(int((max(train_images_class_1_pca) - min(train_images_class_1_pca)) / region_size), 1)
  histogram, bin_edges = np.histogram(train_images_class_1_pca, bins=number_of_bins, density=True)

  # plot the probability distribution for each region size
  plt.subplot(len(region_sizes), 1, region_sizes.index(region_size) + 1)
  plt.bar(bin_edges[:-1], histogram, width=region_size, align='center', alpha=0.5, color='m')
  plt.title(f'Class 1, Region Size: {region_size}')

plt.tight_layout()
plt.show()

# defining the boundaries for uniform distribution based ML classifier
class_0_boundaries = [-1500, -1000, 1000, 1500]
class_1_boundaries = [-999, 999]

predicted_classes_class_0_uniform = []
predicted_classes_class_1_uniform = []

# maximum likelihood classifiers assuming uniform distribution of data (estimated by region size of 1)
for x in train_images_class_0_pca:
  if (class_0_boundaries[0] <= x <= class_0_boundaries[1]) or (class_0_boundaries[2] <= x <= class_0_boundaries[3]):
    predicted_classes_class_0_uniform.append(0)
  elif class_1_boundaries[0] <= x <= class_1_boundaries[1]:
    predicted_classes_class_0_uniform.append(1)
  
for x in train_images_class_1_pca:
  if (class_0_boundaries[0] <= x <= class_0_boundaries[1]) or (class_0_boundaries[2] <= x <= class_0_boundaries[3]):
    predicted_classes_class_1_uniform.append(0)
  elif class_1_boundaries[0] <= x <= class_1_boundaries[1]:
    predicted_classes_class_1_uniform.append(1)

print("\nClass 0 Accuracy assuming uniform distribution: ", (len(predicted_classes_class_0_uniform)-sum(predicted_classes_class_0_uniform))/len(predicted_classes_class_0_uniform))
print("\nClass 1 Accuracy assuming uniform distribution: ", sum(predicted_classes_class_1_uniform)/len(predicted_classes_class_1_uniform))

# defining the mean and variances, will be used for the ML classifier based on quadratic distribution
mean_training_class_0 = np.mean(train_images_class_0_pca)
var_class_0 = np.var(train_images_class_0_pca)
mean_training_class_1 = np.mean(train_images_class_1_pca)
var_class_1 = np.var(train_images_class_1_pca)

predicted_classes_class_0 = []
predicted_classes_class_1 = []

# maximum likelihood classifiers assuming quadratic distribution of data (estimated by region sizes of 10 and 100)
for x in train_images_class_0_pca:
  likelihood_class_0 = (1 / (2 * math.sqrt(var_class_0))) * (math.exp(-((x - mean_training_class_0) ** 2) / (2 * var_class_0)) + math.exp(-((x - mean_training_class_0) ** 2) / (2 * var_class_0)))
  likelihood_class_1 = (1 / (2 * math.sqrt(var_class_1))) * (math.exp(-((x - mean_training_class_1) ** 2) / (2 * var_class_1)) + math.exp(-((x - mean_training_class_1) ** 2) / (2 * var_class_1)))
  if likelihood_class_0 > likelihood_class_1:
    predicted_classes_class_0.append(0)
  else:
    predicted_classes_class_0.append(1)

for x in train_images_class_1_pca:
  likelihood_class_0 = (1 / (2 * math.sqrt(var_class_0))) * (math.exp(-((x - mean_training_class_0) ** 2) / (2 * var_class_0)) + math.exp(-((x - mean_training_class_0) ** 2) / (2 * var_class_0)))
  likelihood_class_1 = (1 / (2 * math.sqrt(var_class_1))) * (math.exp(-((x - mean_training_class_1) ** 2) / (2 * var_class_1)) + math.exp(-((x - mean_training_class_1) ** 2) / (2 * var_class_1)))
  if likelihood_class_0 > likelihood_class_1:
    predicted_classes_class_1.append(0)
  else:
    predicted_classes_class_1.append(1)

print("\nClass 0 Accuracy assuming quadratic distribution: ", (len(predicted_classes_class_0)-sum(predicted_classes_class_0))/len(predicted_classes_class_0))
print("\nClass 1 Accuracy assuming quadratic distribution: ", sum(predicted_classes_class_1)/len(predicted_classes_class_1))

# kernel based estimation
combined_training_data_pca = np.concatenate([train_images_class_0_pca, train_images_class_1_pca])
linspace_kernel_estimate = np.linspace(min(combined_training_data_pca), max(combined_training_data_pca), 1000)
kernel_density = np.zeros(linspace_kernel_estimate.shape)

# size of Gaussian kernel
sigma = 20
# function to calculate kernel
def gaussian_kernel(x, data, sigma):
  return np.sum(np.exp(-(x - data) **2 / (2*sigma**2)) / (sigma * np.sqrt(2 * np.pi)))

# calculating the gaussian kernel for each point of interest
for i in range(linspace_kernel_estimate.shape[0]):
  kernel_density[i] = np.mean(gaussian_kernel(linspace_kernel_estimate[i], combined_training_data_pca, sigma))

# normalizing the kernel density
kernel_density /= (sigma * np.sqrt(2 * np.pi))

plt.tight_layout()
plt.plot(linspace_kernel_estimate, kernel_density, alpha=0.5, color='b')
plt.title("Kernel-Based Density Estimation, sigma=20")
plt.xlabel("Data (after PCA)")
plt.ylabel("Kernel Density")
plt.show()
