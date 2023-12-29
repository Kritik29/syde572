import torch
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import matplotlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("Python Version: ", sys.version)
print("Pytorch Version: ", torch.__version__)
print("Matplotlib Version: ", matplotlib.__version__)

# check if gpu is available and use if available
if torch.cuda.is_available():
  print("CUDA is available!")
  print("CUDA Device Name:", torch.cuda.get_device_name(0))
  device = 'cuda'
else:
  print("CUDA is not available.")
  device = 'cpu'

# 3 Layer MLP implementation
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # MLP layers
    self.layers = nn.Sequential(
      nn.Linear(784, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )

  def forward(self, x):
    # flatten the input to (28x28, 1) = (784, 1)
    x = torch.flatten(x, start_dim=1)
    x = self.layers(x)
    return x

transform = transforms.Compose([
  transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=4096, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)

# use CrossEntropyLoss as loss function and SGD as optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

num_epochs = 5
for epoch in range(num_epochs):
  # training the model
  model.train()
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

  # evaluate trained model
  model.eval()
  correct_train = 0
  total_train = 0
  with torch.no_grad():
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total_train += labels.size(0)
      correct_train += (predicted == labels).sum().item()
  train_accuracy = correct_train / total_train

  # find test accuracy
  correct_test = 0
  total_test = 0
  test_loss = 0
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total_test += labels.size(0)
      correct_test += (predicted == labels).sum().item()
  test_accuracy = correct_test / total_test
  test_losses.append(test_loss / len(test_loader))
  test_accuracies.append(test_accuracy)

  train_losses.append(loss.item())
  train_accuracies.append(train_accuracy)

  print(f'Epoch: {epoch}, Training Accuracy: {train_accuracy}')

epochs_range = range(1, num_epochs + 1)
# plot training and test accuracy
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Test Accuracy over Epochs')
plt.show()

plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.legend()
plt.title('Training Accuracy over Epochs')
plt.show()

# plot training and test loss
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend()
plt.title('Test Loss over Epochs')
plt.show()

plt.plot(epochs_range, train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss over Epochs')
plt.show()
