# q6.2
# ref: https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

data_dir = '../data/oxford-flowers102/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'

training_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # For GPU purpose
    # As we are going to do transfer learning with a ImageNet pretrained VGG
    # so here we normalize the dataset being used here with the ImageNet stats
    # for better transfer learning performance
    transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

testing_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
                         [0.229, 0.224, 0.225])
])

# hyperparams
batch_size = 32
num_epochs = 20
lr = 0.001

# Load the datasets with torchvision.datasets.ImageFolder object
train_data = datasets.ImageFolder(train_dir, transform = training_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform = testing_transforms)

# Define the torch.utils.data.DataLoader() object with the ImageFolder object
# Dataloader is a generator to read from ImageFolder and generate them into batch-by-batch
# Only shuffle during trianing, validation and testing no shuffles
# the batchsize for training and tesitng no need to be the same
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

# architecture is [CONV-POOL-CONV-POOL-FC-FC]
# [batch_size, 3, 224, 224]
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 32, 5)
        self.fc1 = nn.Linear(89888, 2048)
        self.fc2 = nn.Linear(2048, 102)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
conv_net = Net()

np.random.seed(777)
model = conv_net

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    valid_loss = 0
    valid_acc = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        # (batch_size, 36)/ (36, ) 
        inputs, labels = data
        # [32, 3, 224, 224]
        # print(inputs.shape)
        # torch.Size([32])
        # print(labels.shape)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        # (batch_size, 102)
        outputs = model(inputs)
        # print(outputs)
        # print(labels)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)

        loss.backward()
        # Adjust learning weights
        optimizer.step()
        probs, maxidx = torch.max(outputs.data, 1)
        train_acc += ((labels == maxidx).sum().item())
        train_loss += loss.item()
    train_acc = train_acc/len(train_data)
    train_loss = train_loss/len(train_data)

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    for i, data in enumerate(valid_loader):
        # Every data instance is an input + label pair
        # (batch_size, 36)/ (36, ) 
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        probs, maxidx = torch.max(outputs.data, 1)
        valid_acc += ((labels == maxidx).sum().item())
        valid_loss += loss.item()
    valid_acc = valid_acc/len(valid_data)
    valid_loss = valid_loss/len(valid_data)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    # writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, epoch)
    # writer.add_scalars("Acc", {'train': train_acc, 'valid': valid_acc}, epoch)                  
    # writer.flush()
    print("epoch: {}, train_loss: {:.2f}, train_acc: {:.4f}, valid_loss: {:.2f}, valid_acc: {:.4f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))

# test
acc = 0
for i, data in enumerate(test_loader):
    inputs, labels = data[0], data[1]
    outputs = model(inputs)
    for label, prob in zip(labels.detach().numpy(), outputs.detach().numpy()):
        if label == np.argmax(prob):
            acc += 1
print("acc: ", acc/len(test_data))

plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, 'b')
plt.plot(np.arange(len(train_losses)), valid_losses, 'r')
plt.legend(['training loss', 'valid loss'])
plt.show()
plt.figure()
plt.plot(np.arange(len(train_accs)), train_accs, 'b')
plt.plot(np.arange(len(valid_accs)), valid_accs, 'r')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()

# =====squeezenet=====
weights = SqueezeNet1_1_Weights
s_model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
# original settings:
# print(s_model.classifier)
# Sequential(
#   (0): Dropout(p=0.5, inplace=False)
#   (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#   (2): ReLU(inplace=True)
#   (3): AdaptiveAvgPool2d(output_size=(1, 1))
# )
s_model.classifier = nn.Sequential(
  nn.Dropout(p=0.5, inplace=False),
  nn.Conv2d(512, 102, kernel_size=(1, 1), stride=(1, 1)),
  nn.ReLU(inplace=True),
  nn.AdaptiveAvgPool2d(output_size=(1, 1)),
)

model = s_model
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)    

for epoch in range(5):
    train_loss = 0
    train_acc = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        probs, maxidx = torch.max(outputs.data, 1)
        train_acc += ((labels == maxidx).sum().item())
        train_loss += loss.item()
    train_acc = train_acc/len(train_data)
    train_loss = train_loss/len(train_data)
    print("epoch: {}, train_loss: {:.2f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))

for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    valid_loss = 0
    valid_acc = 0    

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        probs, maxidx = torch.max(outputs.data, 1)
        train_acc += ((labels == maxidx).sum().item())
        train_loss += loss.item()
    train_acc = train_acc/len(train_data)
    train_loss = train_loss/len(train_data)

    model.eval()

    for i, data in enumerate(valid_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        probs, maxidx = torch.max(outputs.data, 1)
        valid_acc += ((labels == maxidx).sum().item())
        valid_loss += loss.item()
    valid_acc = valid_acc/len(valid_data)
    valid_loss = valid_loss/len(valid_data)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    print("epoch: {}, train_loss: {:.2f}, train_acc: {:.4f}, valid_loss: {:.2f}, valid_acc: {:.4f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))

# test on squeeze net
acc = 0
model.eval()
for i, data in enumerate(test_loader):
    inputs, labels = data[0], data[1]
    outputs = model(inputs)
    for label, prob in zip(labels.detach().numpy(), outputs.detach().numpy()):
        if label == np.argmax(prob):
            acc += 1
print("acc: ", acc/len(test_data))

plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, 'b')
plt.plot(np.arange(len(train_losses)), valid_losses, 'r')
plt.legend(['training loss', 'valid loss'])
plt.show()
plt.figure()
plt.plot(np.arange(len(train_accs)), train_accs, 'b')
plt.plot(np.arange(len(valid_accs)), valid_accs, 'r')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()