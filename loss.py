import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def DANN(features, ad_net):
    ad_out = ad_net(features)
    print(ad_out.size())
    batch_size = ad_out.size(0) // 2
    #batch_size = 128
    #dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size))
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda(), requires_grad=True)
    dc_target = dc_target.detach()
    dc_target = F.sigmoid(dc_target)
    #dc_target = dc_target.view(ad_out.view(-1))
    print(dc_target.size())
    return nn.BCELoss()(ad_out, dc_target)
