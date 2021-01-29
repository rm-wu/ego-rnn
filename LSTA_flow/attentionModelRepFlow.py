import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
#from MyConvLSTMCell import *
# local
#from .FlowLayer import FlowLayer
#from . import resNetNew
#from .MyConvLSTACell import *

# Colab
from FlowLayer import *
import resNetNew
from MyConvLSTACell import *


class attentionModelRepFlow(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, c_cam_classes=1000):
        super(attentionModelRepFlow, self).__init__()
        self.num_classes = num_classes
        self.resNet = resNetNew.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.flow_layer = FlowLayer(channels=128)
        self.lsta_cell = MyConvLSTACell(512, mem_size, c_cam_classes=c_cam_classes)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, device='cuda'):
        state_att = (torch.zeros(inputVariable.size(1), 1, 7, 7).to(device),
                     torch.zeros(inputVariable.size(1), 1, 7, 7).to(device))
        state_inp = ((torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(device)),
                     (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(device)))
        mid_features = [] # TODO: Change into a PyTorch tensor
        for t in range(inputVariable.size(0)):
            conv3 = self.resNet(inputVariable[t], stage=0)
            mid_features.append(conv3)
        mid_features = torch.stack(mid_features).permute(1, 2, 0, 3, 4)
        flow = self.flow_layer(mid_features)
        flows = flow.permute(2, 0, 1, 3, 4)
        mid_features = mid_features.permute(2, 0, 1, 3, 4)
        for t in range(flows.size(0)):
            logit, feature_conv, x = self.resNet(flows[t] + mid_features[t], stage=1)
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h * w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.resNet.fc.weight[class_idx].unsqueeze(1), feature_conv1).view(x.size(0), 1, 7, 7)
            state_att, state_inp, _ = self.lsta_cell(x, cam, state_att, state_inp)
        feats = self.avgpool(state_inp[0]).view(state_inp[0].size(0), -1)
        logits = self.classifier(feats)
        return logits, feats
