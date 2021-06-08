import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import bilinear_model
from torch.autograd import Variable
import data
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def visualize_feature(x, model, layers=[0, 1,2,3,4]):
    net = nn.Sequential(*list(model.children())[:layers[2]])
    img = net(x)
    transform1 = transforms.ToPILImage(mode='RGB')
    # img = torch.cpu().clone()
    for i in range(img.size(0)):
    #for i in range(1):
        image = img[0]
        print(image.shape)
        print(type(image))
        # print(image.size())
        image = transform1(image)
        image.show()


os.environ['KMP_DUPLICATE_LIB_OK']='True'
image_path='/Users/zhulingang/Desktop/新菊花数据集/test/045.钟山赤焰/IMG_3134.jpeg'
to_tensor=transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ])
img=Image.open(image_path)
img=to_tensor(img)
img = torch.unsqueeze(img, 0)
print(img.shape)
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
visualize_feature(img,model)
model.eval()
outputs=model(img)
outputs = F.softmax(outputs, dim=1)
predicted = torch.max(outputs, dim=1)[1].cpu().item()
print(predicted)
