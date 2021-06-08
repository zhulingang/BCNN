# encoding:utf-8
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import bilinear_model
from torch.autograd import Variable
import data
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
writer1=SummaryWriter('./runs/train_loss')
writer2=SummaryWriter('./runs/test_acc')
trainset = data.MyDataset('/test/Bilinear_CNN-master/CUB200/train.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=0)

testset = data.MyDataset('/test/Bilinear_CNN-master/CUB200/validation.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=0)
cudnn.benchmark = True



model = bilinear_model.Net()
print (model)


pretrained = True
if pretrained:

    pre_dic = torch.load("/Users/zhulingang/Downloads/Bilinear_CNN-master/vgg16-397923af.pth")
    Low_rankmodel_dic = model.state_dict()
    pre_dic = {k: v for k, v in pre_dic.items() if k in Low_rankmodel_dic}
    Low_rankmodel_dic.update(pre_dic)
    model.load_state_dict(Low_rankmodel_dic)




criterion = nn.CrossEntropyLoss()

#特征提取网络所有的输入都不需要保存梯度，那么输出的requires_grad会自动设置为False。既然没有了相关的梯度值，自然进行反向传播时会将这部分子图从计算中剔除
model.features.requires_grad = False


optimizer = optim.SGD([
                       {'params': model.classifiers.parameters(), 'lr': 1.0}], lr=1, momentum=0.9, weight_decay=1e-5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        niter = epoch * len(trainloader) + batch_idx
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        writer1.add_scalars('Last_Train_loss',{'Last_train_loss':loss.data.item()},niter)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    niter = epoch * len(trainloader)
    for data, target in testloader:
        data, target = data, target
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testloader.dataset)
    writer2.add_scalars('Last_Test_acc', {'Last_test_acc': correct}, niter)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 16., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


for epoch in range(1, 81):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    if epoch%5==0:
        test(epoch)
torch.save(model.state_dict(), 'bcnn_lastlayer.pth')
