import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel

        # Implements network described at http://pytorch.org/tutorials/_images/mnist.png
        # TODO: How did they a) arrive at that architecture (arbirtrary numbers?)
        #                    b)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # TODO: what does fc standfor?
        self.fc1   = nn.Linear(16 * 5 * 5, 120)  # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = Variable(torch.range(1, 10))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# print(loss.creator)  # MSELoss
# print(loss.creator.previous_functions[0][0])  # Linear
# print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU
