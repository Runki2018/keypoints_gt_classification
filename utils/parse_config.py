import os
import numpy as np


def parse_model_cfg(path: str):
    """处理模型配置文件，返回一个解析列表，列表元素是定义一个层的字典。"""
    # 检查文件是否存在
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 读取文件信息
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # 去除空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]
    # 去除每行开头和结尾的空格符
    lines = [x.strip() for x in lines]

    md = []  # module definitions
    for line in lines:  # 模型解析列表中添加所有模块元素
        if line.startswith("["):  # this marks the start of a new block
            md.append({})
            md[-1]["type"] = line[1:-1].strip()  # 记录module类型
            # 如果是卷积模块，设置默认不使用BN
            if md[-1]["type"] == "convolutional":
                md[-1]["batch_normalize"] = 0
        else:
            key, val = line.split('=')  # n=4 -> key=n, val=4
            key, val = key.strip(), val.strip()

            if str_isnumeric(val):  # 如果是数值的情况
                md[-1][key] = int(val) if (round(float(val)) - float(val)) == 0 else float(val)
            else:
                md[-1][key] = val  # 字符串的情况

    # check all fields are supported  每个模块字典中所支持的关键字
    supported = ['type', 'batch_normalize', 'out_features', 'activation', 'dim']

    # 遍历检查每个模型的配置
    for x in md[1:]:  # 0对应net配置
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))

    return md


def str_isnumeric(val: str):
    """判断字符串内容是否为数值（int or float）"""
    val_list = val.split('.')
    for val in val_list:
        if not val.isnumeric():
            return False
    return True
