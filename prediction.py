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



testset = data.MyDataset('/test/Bilinear_CNN-master/CUB200/test.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)
model = bilinear_model.Net()
pretrained = True
if pretrained:
    pre_dic = torch.load('bcnn_alllayer.pth')
    model.load_state_dict(pre_dic)
else:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = data, target
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 16., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))
test()