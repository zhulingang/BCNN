import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# vgg16 = torchvision.models.vgg16(pretrained=True)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


num_class=115
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),


            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),



            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),


            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )

        self.classifiers = nn.Sequential(
            nn.Linear(512 ** 2, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 512, 28 ** 2)  #??????batch_size,channel ,h,w????????????batch_size,channel,h*w????????????
        #torch.transpose(x, 1, 2))????????????????????????1???2?????????
        #torch.bmm????????? torch.bmm(a,b),tensor a ???size???(b,h,w),tensor b???size???(b,w,h),????????????tensor??????????????????3. ??????(batch_szie,512,512)
        #/ 28 ** 2????????????
        #view(batch_size, -1)?????????????????????
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 28 ** 2)
        print(x.shape)
        x=x.view(batch_size, -1)


        #normalize?????????
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        # feature = feature.view(feature.size(0), -1)
        x = self.classifiers(x)
        return x
