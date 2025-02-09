import torch
import torch as nn


class C3D(nn.Module):
    def  __init__(self,num_classes, pretraine=False):
        super(C3D, self).__init()
        self.conv1 = 