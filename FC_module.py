import torch
from torch import nn


class MyFC(nn.Module):
    def __init__(self, layer, d_in, d_out):
        super(MyFC, self).__init__()  # 固定语句，继承nn.Module构造函数
        # layer_list = [60, 40, 20]  # acc =  0.65
        self.fc = nn.Sequential(
            nn.Linear(d_in, layer[0]),
            nn.LeakyReLU(),
            nn.Linear(layer[0], layer[1]),
            nn.LeakyReLU(),
            nn.Linear(layer[1], layer[2]),
            nn.LeakyReLU(),
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
        inc = 0.01
        for op in self.modules():
            # 针对不同类型操作采用不同初始化方式
            if isinstance(op, nn.Linear):
                print("weight init!!")
                # nn.init.constant_(op.weight.data, val=2)
                nn.init.normal_(op.weight, mean=0, std=0.1)
                nn.init.normal_(op.weight, mean=0, std=inc)
                inc *= 10  # 观察学的好的网络，越后的FC权重绝对值越大
            else:  # 这里可以对Conv等操作进行其它方式的初始化
                pass


if __name__ == '__main__':
    layer_fc = [256, 128, 64]
    model = MyFC(layer_fc, 42, 8)
    param_path = "./runs/2021-03-25/acc80_epoch267_256_128_256"
    state = torch.load(param_path)["model_state"]
    model.load_state_dict(state)
    print(model.state_dict())
