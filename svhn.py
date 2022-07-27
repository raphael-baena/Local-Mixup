import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
import numpy as np

root = './data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fashion_mnist(batch_size = 64):
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data                                                                                             
  trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
  testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

  return train_loader, test_loader

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out
class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(1, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x,target = None,mixup = None):
        if mixup is None:
          target_a, target_b, lam = None, None, None
          dist = torch.ones(x.size()[0]).cuda()

        else:
          x,target_a,target_b,lam ,dist= mixup_data(x,target)
          "dist = torch.ones(x.size()[0]).cuda()"
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, target_a, target_b, lam ,torch.ones(x.size()[0]).cuda()
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    sig = 0.0005
    dist = torch.exp( -sig * torch.sqrt(( (x.flatten(start_dim = 1) - x[index, :].flatten(start_dim = 1)) **2).sum(dim=1)) )

    return mixed_x, y_a, y_b, lam,dist

def mix_criterion(criterion, pred, y_a, y_b,lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

def train(model, train_loader, optimizer,mixup = None):
    model.train()
    accuracy, total_loss = 0., 0.
    for idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,y_a,y_b,lam,dist = model(data,target = target,mixup = mixup)
        if mixup is None:
            #print("mixup is none")                                                                                                                                                   \
                                                                                                                                                                                       
            loss = criterion(output, target)
        else:
            loss  = mix_criterion(criterion, output, y_a.flatten().long(), y_b.flatten().long(), lam)
        loss = (loss*dist).sum() / dist.sum()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return { "train_loss" : total_loss / len(train_loader)}
def train_era(model, epochs, loaders,mixup = None):
    train_loader, test_loader = loaders
    for epoch in range(epochs):
        train_stats = train(model, train_loader, optimizer,mixup = mixup)
        scheduler.step()
        test_stats = test(model, test_loader)
        print("\rEpoch: {:3d}, test_acc: {:.2f}%, train_loss: {:.5f}".format(epoch, 100 * test_stats["test_acc"], train_stats["train_loss"]), end = "")
    print()
    return test_stats["test_acc"]

criterion = nn.CrossEntropyLoss(reduction = 'none')

net = densenet_cifar()
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

def test(model, test_loader):
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,__,__,__,__ = model(data)
            test_loss += criterion(output, target).mean().item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
    return { "test_loss" : test_loss / len(test_loader), "test_acc" : accuracy / len(test_loader.dataset) }


epochs = 40
loaders =  fashion_mnist()
train_era(net, epochs, loaders,mixup = True)