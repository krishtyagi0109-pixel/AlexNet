import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets,transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

from torchvision.datasets import ImageNet

from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = transform
)

train_loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle=True
)

from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
class AlexNet(nn.Module):

  def __init__(self):
    super().__init__()



    self.features = nn.Sequential(
        nn.Conv2d(3,96,kernel_size=11,stride = 4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride=2),

        nn.Conv2d(96,256,kernel_size=5,padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride=2),

        nn.Conv2d(256,384,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384,384,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384,256,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride =2)



    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256*6*6,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,10)
    )

  def forward(self,x):
    x = self.features(x)
    x = self.fc(x)
    return x


image,label = next(iter(train_loader))

model = AlexNet().to(device)

output = model(image.to(device))

criterion = nn.CrossEntropyLoss()

loss = criterion(output,label.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=  0.0001)

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

print("Loss:", loss.item())
print("Gradient:", model.features[0].weight.grad.abs().mean().item())

epochs = 10

for epoch in range(epochs):
  tot_loss = 0
  for images,labels in train_loader:
    images,labels = images.to(device),labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    tot_loss += loss.item()

    avg_loss = tot_loss/len(train_loader)



  print(f"Epoch {epoch+1}: {avg_loss:.4f}")



#TEST

import matplotlib.pyplot as plt


test_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor()

])

test_set = CIFAR10(root = "./data", train = False, download=True, transform=test_transform)

test_loader = DataLoader(test_set, batch_size = 128, shuffle=True)

classes = test_set.classes

model.eval()
correct,total = 0,0

with torch.no_grad():
  for images , labels in test_loader:
    images,labels = images.to(device), labels.to(device)
    preds = model(images).argmax(dim = 1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)


print(f"accuracy:-{(correct/total)*100}")

images, labels = next(iter(test_loader))
with torch.no_grad():
    preds = model(images.to(device)).argmax(dim=1).cpu()

plt.figure(figsize=(12,6))
for i in range(12):
  plt.subplot(3,4,i+1)
  plt.imshow(images[i].permute(1,2,0))
  color = "green" if preds[i] == labels[i] else "red"
  plt.title(f"Pred:{classes[preds[i]]}\n {classes[labels[i]]}", color=color, fontsize=9)
  plt.axis("off")

plt.tight_layout()

plt.show()

