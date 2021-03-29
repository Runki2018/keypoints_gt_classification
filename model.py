from torch import nn
from utils.parse_config import parse_model_cfg


def create_modules(modules_definitions : list):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_definitions:  通过.cfg文件解析得到的每个层结构的列表
    :return: a nn.ModuleList
    """
    net_block = modules_definitions.pop(0)  # cfg training hyperparams (unused) , pop [net] block
    in_features = net_block["in_features"]  # in_features for the next full connection layer
    module_list = nn.ModuleList()  # ModuleList include the network layer

    # 遍历搭建每个层结构
    for i, block in enumerate(modules_definitions):
        modules = nn.Sequential()  # 每层的结构存储于一个网络序列容器中

        if block["type"] == "fc":  # 1、fc
            out_features = block["out_features"]
            if isinstance(out_features, int):
                fc = nn.Linear(in_features=in_features, out_features=out_features)
                # init weight and bias
                # nn.init.constant_(op.weight.data, val=2)
                nn.init.normal_(fc.bias, mean=0, std=0.01)
                nn.init.normal_(fc.weight, mean=0, std=0.01)  # 观察了一下发现权重基本上是0.01左右
                modules.add_module("fc", fc)
                in_features = out_features
            else:
                raise TypeError("fc out_features must be int type")

            bn = block["batch_normalize"]  # 1 or 0 / use or not
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm1d(in_features))

            if block["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            elif block["activation"] == "relu":
                modules.add_module("activation", nn.ReLU(inplace=True))
            else:
                pass

        elif block["type"] == "softmax":  # 2、softmax
            dim = block["dim"]
            modules.add_module("softmax", nn.Softmax(dim=dim))

        else:
            pass

        # Register module list and number of output filters
        module_list.append(modules)
    return module_list


class classifier(nn.Module):
    def __init__(self, cfg: str):
        super(classifier, self).__init__()
        # 解析网络对应的.cfg文件,得到模型定义
        self.module_definitions = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建
        self.module_list = create_modules(self.module_definitions)

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__  # 可以通过 name 来对不同层进行操作，如 name == "Sequential"
            x = module(x)
        return x
