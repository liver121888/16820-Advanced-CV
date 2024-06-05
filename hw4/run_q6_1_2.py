# q6.1.1
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class NIST36_Data(torch.utils.data.Dataset):
    def __init__(self, type):
        self.type = type
        self.data = scipy.io.loadmat(f"../data/nist36_{type}.mat")
        self.inputs, self.one_hot_target = (
            self.data[f"{self.type}_data"],
            self.data[f"{self.type}_labels"],
        )
        self.target = np.argmax(self.one_hot_target, axis=1)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs[index]).type(torch.float32)
        target = torch.tensor(self.target[index]).type(torch.LongTensor)
        return inputs, target
    
# ref: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        # print(x.shape)
        x = self.conv2(x)
        x = self.pool(torch.relu(x))
         # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
conv_net = Net()

train_data = NIST36_Data(type="train")
valid_data = NIST36_Data(type="valid")
test_data = NIST36_Data(type="test")

batch_size = 64
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# reference: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

model = conv_net
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

num_epochs = 100
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

np.random.seed(777)

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    valid_loss = 0
    valid_acc = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainloader):
        # Every data instance is an input + label pair
        # (batch_size, 36)/ (36, ) 
        inputs, labels = data
        inputs = inputs.reshape(-1, 1, 32, 32)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        # (batch_size, 36)
        outputs = model(inputs)
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

    for i, data in enumerate(validloader):
        # Every data instance is an input + label pair
        # (batch_size, 36)/ (36, ) 
        inputs, labels = data
        inputs = inputs.reshape(-1, 1, 32, 32)
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

# test
acc = 0
for i, data in enumerate(testloader):
    inputs, labels = data[0], data[1]
    inputs = inputs.reshape(-1, 1, 32, 32)
    outputs = model(inputs)
    for label, prob in zip(labels.detach().numpy(), outputs.detach().numpy()):
        if label == np.argmax(prob):
            acc += 1
print("acc: ", acc/len(test_data))

plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, 'b')
plt.plot(np.arange(len(valid_losses)), valid_losses, 'r')
plt.legend(['training loss', 'valid loss'])
plt.show()
plt.figure()
plt.plot(np.arange(len(train_accs)), train_accs, 'b')
plt.plot(np.arange(len(valid_accs)), valid_accs, 'r')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()