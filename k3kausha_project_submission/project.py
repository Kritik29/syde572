import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.models as models
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

# print dependency information and CUDA information
print("\n")
print("Python Version: ", sys.version)
print("Pytorch Version: ", torch.__version__, "\n")

if torch.cuda.is_available():
  print("CUDA is available!")
  print("CUDA Device Name:", torch.cuda.get_device_name(0), "\n")
  device = 'cuda'
else:
  print("CUDA is not available.")
  device = 'cpu'

# define transformation, will be used for both train and test set
train_transform = transforms.Compose([
  transforms.Resize((360, 360)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])

# load the training images
data_path = './5_shot/5_shot/train'

train_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# use a pretrained ResNet18 model with default weights
rn18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# fully connected layer
rn18.fc = nn.Sequential(
  nn.Linear(rn18.fc.in_features, 22),
)

# define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(rn18.parameters(), lr = 0.0000875)

# send model to GPU if available
rn18.to(device)

num_epochs = 35

# training the model with fine-tuned hyper parameters
for epoch in range(num_epochs):
  running_loss = 0.0
  correct_predictions = 0
  total_predictions = 0
  rn18.train()

  for inputs, labels in train_loader:
    # send images and labels to GPU if available
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = rn18(inputs)
    loss = loss_function(outputs, labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)

    _, predicted = torch.max(outputs, 1)
    total_predictions += labels.size(0)
    correct_predictions += (predicted == labels).sum().item()

  epoch_loss = running_loss / len(train_loader.dataset)
  epoch_accuracy = correct_predictions / total_predictions

  print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy * 100:.2f}%')

# load test data
test_data = []
test_data_path = './5_shot/5_shot/test'
test_files = os.listdir(test_data_path)

# loop through test folder and add image data and ID to test_data
for file in test_files:
  image_path = os.path.join(test_data_path, file)
  input_image = Image.open(image_path).convert('RGB')
  input_image = train_transform(input_image)
  test_data.append((input_image, file))

test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

rn18.eval()

# use the same class-id map as in training to ensure consistency
# create a map to translate between Pytorch's predicted IDs and our actual class labels
class_id_map = train_dataset.class_to_idx
id_class_map = {value: key for key, value in class_id_map.items()}

predictions = []
image_ids = []

# make predicitions on the test data
with torch.no_grad():
  for image, filename in test_loader:
    image = image.to(device)
    outputs = rn18(image)
    _, predicted = torch.max(outputs.data, 1)
    
    # add predictions and ID to their arrays
    predictions.append(id_class_map[predicted.item()])
    image_ids.append(filename[0].split('.')[0])

# combine IDs and predicitions into a list of tuples, sorted by the ID
sorted_predictions = sorted(zip(image_ids, predictions), key=lambda x: int(x[0]))

# write the IDs and predictions into a CSV
with open('k3kausha_predictions.csv', 'w') as file:
  file.write('id,category\n')
  for image_id, predicted_label in sorted_predictions:
    file.write(f'{image_id},{predicted_label}\n')
