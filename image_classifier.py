# simple_image_classifier.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# Set up image transformations
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64 pixels
    transforms.ToTensor()         # Convert image to PyTorch tensor
])

# -----------------------------------------
# Load training and test datasets
# Folder structure should be:
# train/class_name/image.jpg
# test/class_name/image.jpg
# -----------------------------------------
train_data = datasets.ImageFolder('train/', transform=transform)
test_data = datasets.ImageFolder('test/', transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# -----------------------------------------
# Define a simple feedforward neural network
# -----------------------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)  # Input size: 64*64*3 = 12288
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)      # Binary classification (2 output classes)

    def forward(self, x):
        x = x.view(-1, 12288)            # Flatten input
        x = F.relu(self.fc1(x))          # First hidden layer + ReLU
        x = F.relu(self.fc2(x))          # Second hidden layer + ReLU
        x = self.fc3(x)                  # Output layer (logits)
        return x

# -----------------------------------------
# Instantiate model, define loss and optimizer
# -----------------------------------------
net = SimpleNet()
criterion = nn.CrossEntropyLoss()                           # Suitable for multi-class classification
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# -----------------------------------------
# Training loop
# -----------------------------------------
for epoch in range(5):  # Number of training epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()             # Zero the parameter gradients
        outputs = net(inputs)            # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                  # Backward pass
        optimizer.step()                 # Optimize weights

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print("Finished Training")

# -----------------------------------------
# Evaluate on test set
# -----------------------------------------
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for evaluation
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test images: {100 * correct / total:.2f}%")

# -----------------------------------------
# Visualize some training images
# -----------------------------------------
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize if needed
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of training data and show the first few images
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:4]))
print('Labels:', ' '.join(f'{train_data.classes[labels[j]]}' for j in range(4)))
