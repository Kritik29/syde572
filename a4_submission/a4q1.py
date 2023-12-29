import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torchvision.datasets import MNIST

#############################################################################################

# Exercise 1

np.random.seed(10)

# data import and preprocessing below

mnist_training_data = MNIST(root='./data', train=True, download=True, transform=None)
mnist_test_data = MNIST(root='./data', train=False, download=True, transform=None)

# since we are instructed to only use the first two classes, we will only select images and labels
# from those classes
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
train_flattened = np.vstack([train_images_class_0_flattened, train_images_class_1_flattened])

# perform dimension reduction with PCA to get 2x1 vectors
pca = PCA(n_components=2)
training_data = pca.fit_transform(train_flattened).astype(np.float64)

training_labels = np.concatenate([train_labels_class_0, train_labels_class_1]).astype(int)

# normalize data and convert labels to int array
training_data = np.array(training_data) / 255.0
training_labels = np.array(training_labels).astype(int)

# load up the test data
class_0 = (0)
subset_idx_0 = torch.isin(mnist_test_data.targets, torch.as_tensor(class_0))
test_images_class_0 = mnist_test_data.data[subset_idx_0].numpy()
test_labels_class_0 = mnist_test_data.targets[subset_idx_0].numpy()

class_1 = (1)
subset_idx_1 = torch.isin(mnist_test_data.targets, torch.as_tensor(class_1))
test_images_class_1 = mnist_test_data.data[subset_idx_1].numpy()
test_labels_class_1 = mnist_test_data.targets[subset_idx_1].numpy()

test_images_class_0_flattened = test_images_class_0.reshape(-1, 28*28)
test_images_class_1_flattened = test_images_class_1.reshape(-1, 28*28)
test_flattened = np.vstack([test_images_class_0_flattened, test_images_class_1_flattened])

test_data = pca.fit_transform(test_flattened).astype(np.float64)
test_labels = np.concatenate([test_labels_class_0, test_labels_class_1])

test_data = np.array(test_data) / 255.0
test_labels = np.array(test_labels).astype(int)

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

w = np.random.randn(training_data.shape[1])
w0 = 0

learning_rate = 0.1
num_epochs = 100

losses = []
accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
  # calculate the "z" to feed into the sigmoid function for both train and test sets
  z = np.dot(training_data, w) + w0
  z_test = np.dot(test_data, w) + w0

  p_c1 = sigmoid(z)
  predictions_test = sigmoid(z_test)

  # calculate training/test losses and store in array
  loss = -np.mean(training_labels * np.log(p_c1) + (1 - training_labels) * np.log(1 - p_c1))
  losses.append(loss)

  test_loss = -np.mean(test_labels * np.log(predictions_test) + (1 - test_labels) * np.log(1 - predictions_test))
  test_losses.append(test_loss)

  # calculate training/test accuracy and store error (1-accuracy) in array
  predictions = (p_c1 >= 0.5).astype(int)
  accuracy = np.mean(predictions == training_labels)
  accuracies.append(1-accuracy)

  predictions_test = (predictions_test >=0.5).astype(int)
  test_accuracy = np.mean(predictions_test == test_labels)
  test_accuracies.append(1-test_accuracy)

  # calculate gradients
  gradient_w = np.dot(training_data.T, (p_c1 - training_labels)) / len(training_labels)
  gradient_w0 = np.sum(p_c1 - training_labels) / len(training_labels)

  # update parameters based on gradients
  w -= learning_rate * gradient_w
  w0 -= learning_rate * gradient_w0

  print(f"Epoch: {epoch}, Test Accuracy: {test_accuracy}")

# plot various errors and losses against the number of epochs
plt.plot(test_accuracies, label='Test Error')
plt.title('Test Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

plt.plot(accuracies, label='Train Error')
plt.title('Training Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

plt.plot(test_losses, label='Test Loss')
plt.title('Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(losses, label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
