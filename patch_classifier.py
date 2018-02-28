'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

import util.image_related as image_rel

class VGG_Choi(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG_Choi, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            #nn.Linear(512, 10),# for 0 to 9
            nn.Linear(512, 2),  # for cancer or normal
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        #print (x.size())
        # cfg_E : [1, 512, 15, 15], cfg_A : [1, 512, 15, 15], cfg_F : [1, 512, 7, 7], cfg_G : [1, 512, 3, 3], cfg_H : [1, 512, 1, 1]
        x = x.view(x.size(0), -1)
        #print (x.size())
        x = self.classifier(x)
        #print (x.size())
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

cfg_A = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
cfg_E = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M']
cfg_F = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M']
cfg_G = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M']
cfg_H = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M']


def vgg19(w_h_input_img, batch_norm):
    """VGG 19-layer model (configuration "E")"""
    return VGG_Choi(make_layers(
        #cfg_E,
        #cfg_A,
        #cfg_F,
        #cfg_G,
        cfg_H,
        batch_norm))



def probe_patch_classifier(model, w_h_input_img):
    im_rgb = image_rel.generate_random_image(w_h_input_img)
    a_im_rgb = np.array(im_rgb)
    #a_im_rgb = a_im_rgb[np.newaxis, :, :]
    transform = transforms.Compose([transforms.ToTensor()])
    t_im_rgb = transform(a_im_rgb)
    #t_im_rgb = torch.from_numpy(a_im_rgb)
    #print (t_im_rgb.size())
    t_im_rgb = t_im_rgb.unsqueeze(0)
    #print (t_im_rgb.size())
    v_im_rgb = Variable(t_im_rgb, requires_grad=True)
    out_put = model(v_im_rgb)
    a_out_put = out_put.data.cpu().numpy()
    #print (a_out_put)
    for (x, y), value in np.ndenumerate(a_out_put):
        print (x, y, value)
        if value is None:
            return False
    return True

def create_patch_classifier(w_h_input_img, use_bn, use_gpu):
    model = vgg19(w_h_input_img, use_bn)
    if use_gpu:
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    is_ok = probe_patch_classifier(model, w_h_input_img)
    if is_ok is False:
        model = None
    return model
