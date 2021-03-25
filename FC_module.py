import torch
from torch import nn


class MyFC(nn.Module):
    def __init__(self, layer, d_in, d_out):
        super(MyFC, self).__init__()  # 固定语句，继承nn.Module构造函数
        # layer_list = [60, 40, 20]  # acc =  0.65
        self.fc = nn.Sequential(
            nn.Linear(d_in, layer[0]),
            nn.ReLU(),
            nn.Linear(layer[0], layer[1]),
            nn.ReLU(),
            nn.Linear(layer[1], layer[2]),
            nn.ReLU(),
            nn.Linear(layer[2], d_out),
        )
        self.softmax = nn.Softmax(dim=1)  # (batch, 1, 9)
        self.weight_init()

    def forward(self, x):
        # out = self.fc(x).unsqueeze(dim=1)
        out = self.fc(x)
        out = self.softmax(out)  # (batch, classes)
        return out

    def weight_init(self):
        # 递归获得net的所有子代Module
        for op in self.modules():
            # 针对不同类型操作采用不同初始化方式
            if isinstance(op, nn.Linear):
                print("weight init!!")
                # nn.init.constant_(op.weight.data, val=2)
                nn.init.constant_(op.bias, val=0)
                nn.init.normal_(op.weight, mean=0, std=0.01)
            else:  # 这里可以对Conv等操作进行其它方式的初始化
                pass


