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
import cv2
import random
from torchvision.utils import make_grid

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(img,model):
    # pic_dir = './images/2.jpg'
    # transform = transforms.ToTensor()
    # img = get_picture(pic_dir, transform)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # 插入维度
    # img = img.unsqueeze(0)
    #
    # img = img.to(device)

    # net = models.resnet101().to(device)
    # net.load_state_dict(torch.load('./model/resnet101-5d3b4d8f.pt'))
    exact_list = None
    dst = './feautures'
    therd_size = 256

    myexactor = FeatureExtractor(model, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        if 'fc' in k:
            continue
        features_t=features.unsqueeze(0).permute((1, 0, 2, 3))
        print(features_t.shape)
        im_all=make_grid(features_t, nrow=int(math.sqrt(features_t.size(0))),padding=0).permute((1, 2, 0))
        im_all=(im_all.data.numpy()*255.).astype(np.uint8)
        im_all = cv2.applyColorMap(im_all, cv2.COLORMAP_JET)
        Image.fromarray(im_all).save(dst+'/'+str(k) + '.jpg')
        # for i in range(iter_range):
        #     # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
        #     if 'fc' in k:
        #         continue
        #
        #     feature = features.data.numpy()
        #     feature_img = feature[i, :, :]
        #     feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        #
        #     dst_path = os.path.join(dst, k)
        #
        #     make_dirs(dst_path)
        #     feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        #     if feature_img.shape[0] < therd_size:
        #         tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
        #         tmp_img = feature_img.copy()
        #         tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
        #         cv2.imwrite(tmp_file, tmp_img)
        #
        #     dst_file = os.path.join(dst_path, str(i) + '.png')
        #     cv2.imwrite(dst_file, feature_img)
        for i in range(1):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            feature = features.data.numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            dst_file = os.path.join(dst_path, k + '_' + str(i) + '.png')
            if feature_img.shape[0] < therd_size:
            #    tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(dst_file, tmp_img)
            else:
                cv2.imwrite(dst_file, feature_img)



os.environ['KMP_DUPLICATE_LIB_OK']='True'
image_path='/Users/zhulingang/Desktop/新菊花数据集/test/045.钟山赤焰/IMG_3134.jpeg'
to_tensor=transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ])
img=Image.open(image_path)
#plt.imshow(img)
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
model.eval()
#TODO 画双线性融合特征图
b1=model.features(img)
batch_size = b1.size(0)
b1 = b1.view(batch_size, 512, 28 ** 2)
b1 = (torch.bmm(b1, torch.transpose(b1, 1, 2)) / 28 ** 2)
print(b1.shape)
b1=b1.view(batch_size, -1)
#normalize标准化
b1 = torch.sign(b1) * torch.sqrt(torch.abs(b1) + 1e-10)
b1 = b1.view(batch_size, 512, -1)
b1=b1.squeeze(0)
print(b1.shape)
print(b1)
b1_img=(b1.data.numpy()*255.).astype(np.uint8)
b1_img = cv2.applyColorMap(b1_img, cv2.COLORMAP_JET)
Image.fromarray(b1_img).save('bilinear'+ '.jpg')

#TODO 画卷积层图
get_feature(img,model.features)

outputs=model(img)
outputs = F.softmax(outputs, dim=1)
x=[ i for i in range(115)]
print(np.array(x).shape)
outputs_list=outputs.data.squeeze(0).numpy()
plt.bar( x=x,height=outputs_list,linewidth=1)
plt.show()


print(outputs.data.numpy().shape)
predicted = torch.max(outputs, dim=1)[1].cpu().item()
print(predicted)
