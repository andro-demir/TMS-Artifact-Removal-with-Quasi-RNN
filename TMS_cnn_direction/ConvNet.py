import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Takes 1 input channel, returns 96 output channels, 
        # uses square kernels 3x3 and stride is 1:
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
        self.pool = nn.MaxPool2d(3, stride=2)
        # Takes 6 input channels, returns 16 output channels,
        # uses square kernels 3x3 and stride is 1:
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Takes 256 * 6 * 6 input channels and returns 4096:
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)
        # Accelerate training by reducing internal covariate shift
        # with batch normalization:
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv5_drop = nn.Dropout2d(p=0.2)
        self.fc1_drop = nn.Dropout2d(p=0.2)       

    def forward(self, x):
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5_drop(self.pool(F.relu(self.conv5(x))))
        # Flatten the output tensor before it goes to the FC layer:
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def train(model, device, train_data, criterion, optimizer, epoch):
    # Set the module in training mode:
    model.train()
    for batch_idx, (input, output) in enumerate(train_data, 1):
        input, output = input.to(device), output.to(device)
        # Gradients of all params = 0:
        optimizer.zero_grad()
        # forward + backward + optimize
        pred = model(input)
        loss = criterion(pred, output)
        loss.backward()
        optimizer.step()
        # print statistics
        if batch_idx % 2 == 0: # Print every 2 minibatches 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # Saves the trained model:
    filepath = "C:/Users/ADemir/Desktop/Workspace/TMS"
    torch.save(model.state_dict(), filepath)

def test(model, device, criterion, test_data):
    # Set the module in evaluate mode:
    model.eval()
    test_loss = 0
    # torch.no_grad() disables gradient calculation 
    # It is useful for inference. Since we are not doing backprop in testing,
    # it reduces memory consumption for computations that would otherwise 
    # have requires_grad=True.
    with torch.no_grad():
        for input, output in test_data:
            input, output = input.to(device), output.to(device)
            pred = model(input)
            # 
            test_loss += criterion(pred, output, size_average=False).item() 

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))





