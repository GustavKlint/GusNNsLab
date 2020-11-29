
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#This is a branch
class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < -4] = 0
        return grad_x

def plot_grad_flow(named_parameters, epoch, batch_idx, model, imageCounter):
    # Plots the gradients flowing through different layers in the net during training.
    # Can be used for checking for possible gradient vanishing / exploding problems.
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    
    if imageCounter == 0:
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)

    line = plt.plot(ave_grads, alpha=0.3, color="r")
    if imageCounter % 50 == 0:
        plt.savefig('epoch{}batch{}number{}.png'.format(epoch, batch_idx, imageCounter))
    plt.setp(line[0], color = 'black')
    imageCounter += 1
    return imageCounter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        #self.conv3 = nn.Conv2d(64, 128, 3, 1)
        #self.conv4 = nn.Conv2d(128, 128, 3, 1)
        #self.conv5 = nn.Conv2d(128, 128, 3, 1)
        #self.conv6 = nn.Conv2d(128, 256, 3, 1)
        #self.conv7 = nn.Conv2d(256, 256, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        #self.dropout3 = nn.Dropout2d(0.75)
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, 10)
        #self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        #x = FakeReLU.apply(x)
        x = F.relu(x)
        #print(x.size())

        x = self.conv2(x)
        #x = FakeReLU.apply(x)
        x = F.relu(x)
        #print(x.size())

        #x = self.conv3(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)
        #print(x.size())

        #x = self.conv4(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)
        #print(x.size())

        #x = self.conv5(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)
        #print(x.size())

        #x = self.conv6(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)
        #print(x.size())

        #x = self.conv7(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)
        #print(x.size())

        x = F.max_pool2d(x, 2)
        #print(x.size())
        x = self.dropout1(x)
        #print(x.size())

        x = torch.flatten(x, 1)
        #print(x.size())

        x = self.fc1(x)        
        #x = FakeReLU.apply(x)
        x = F.relu(x)
        #print(x.size())
        #x = self.dropout2(x)

        x = self.fc2(x)
        #x = FakeReLU.apply(x)
        #x = F.relu(x)

        #x = self.dropout3(x)

        #x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    imageCounter = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        #imageCounter = plot_grad_flow(model.named_parameters(), epoch, batch_idx, model, imageCounter)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', default = './tiny-imagenet-200',
                        help='path to dataset')                 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        pin_memory=True)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()