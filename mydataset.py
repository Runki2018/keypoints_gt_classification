import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils.parse_config import parse_model_cfg


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, samples, mode='train', n_classes=9):
        self.label_list = samples[0]  # list
        self.keypoints_list = samples[1]  # list
        self.mode = mode
        # self.transform = transform
        self.n_classes = n_classes

    def __len__(self):
        """数据集样本数"""
        return len(self.label_list)

    def __getitem__(self, idx):
        """每次返回一个样本的标签和关键点"""
        keypoints = torch.tensor(self.keypoints_list[idx])  # (1,42)
        if self.mode == 'train' and torch.rand(1) > 0.5:
            keypoints = self.translation(keypoints)
        # label = torch.zeros(9, dtype=torch.float)  # one-hot code for MSELoss
        # label[self.label_list[idx]] = 1  # one-hot code for MSELoss
        if self.n_classes == 8:  # for CrossEntropyLoss
            label = torch.tensor(self.label_list[idx], dtype=torch.long)-1
        else:
            label = torch.tensor(self.label_list[idx], dtype=torch.long)
        if self.mode == "total":
            return idx, label, keypoints  # 通过idx确定错分样本
        else:
            return label, keypoints

    @staticmethod
    def translation(keypoints):
        # x轴偏移量
        if torch.rand(1) < 0.5:
            x_max = keypoints[..., ::2].max()  # max([x0, x1, ..., x20])
            offset = (1 - x_max) * torch.rand(1)  # 0~1 向右偏移量
            keypoints[..., ::2] += offset
        else:
            x_min = keypoints[..., ::2].min()  # max([x0, x1, ..., x20])
            offset = x_min * torch.rand(1)  # 0~1 向左偏移量
            keypoints[..., ::2] -= offset
        # y轴偏移量
        if torch.rand(1) < 0.5:
            y_max = keypoints[..., 1::2].max()  # max([x0, x1, ..., x20])
            offset = (1 - y_max) * torch.rand(1)  # 0~1 向右偏移量
            keypoints[..., 1::2] += offset
        else:
            y_min = keypoints[..., 1::2].min()  # max([x0, x1, ..., x20])
            offset = y_min * torch.rand(1)  # 0~1 向左偏移量
            keypoints[..., 1::2] -= offset
        return keypoints


class MyDataLoader:
    """数据加载器，用于加载自定义的数据集"""

    def __init__(self, batch_size=19):
        cfg_path = "./cfg/network.cfg"
        net_block = parse_model_cfg(cfg_path)[0]  # [net]
        self.n_classes = net_block["n_classes"]
        self.train_list = read_txt(net_block["train_set"])
        self.test_list = read_txt(net_block["test_set"])
        self.total_list = read_txt(net_block["total_set"])
        self.BATCH_SIZE = batch_size  # 一次读入多少个样本
        self.num_workers = 2  # 加载batch的线程数

    def train(self):
        training_set = MyDataSet(samples=self.train_list, mode="train", n_classes=self.n_classes)
        train_loader = DataLoader(dataset=training_set, batch_size=self.BATCH_SIZE,
                                  shuffle=True, num_workers=self.num_workers)
        return train_loader

    def test(self):
        test_set = MyDataSet(samples=self.test_list, mode="test", n_classes=self.n_classes)
        test_loader = DataLoader(dataset=test_set, batch_size=self.BATCH_SIZE,
                                 shuffle=True, num_workers=self.num_workers)
        return test_loader

    def total_analysis(self):
        total_set = MyDataSet(samples=self.total_list, mode="total")

        total_loader = DataLoader(dataset=total_set, batch_size=self.BATCH_SIZE,
                                  shuffle=False, num_workers=1)
        return total_loader


def read_txt(file_name: str):
    """读取txt文件中的label 和 keypoints"""
    label_list, keypoints_list = [], []
    assert os.path.exists(file_name), "{} :file not exist!".format(file_name)
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]  # "label x1 y1 ... x21 y21\n"
            line = line.split(' ')  # [label, x1, y1, ..., x21, y21]
            keypoints = []
            for i, x in enumerate(line):
                if i == 0:
                    label_list.append(int(x))
                else:
                    keypoints.append(float(x))
            keypoints_list.append(keypoints)
    classes = ['0-其他', '1-OK', '2-手掌', '3-向上', '4-向下', '5-向右', '6-向左', '7-比心', '8-嘘']
    count = [0 for _ in range(9)]
    for x in label_list:
        count[x] += 1
    for i in range(9):
        print("类别 {}\t的数量为 {} ".format(classes[i], count[i]))
    return label_list, keypoints_list


if __name__ == '__main__':
    b = 20
    loader = MyDataLoader(batch_size=b).test()
    for i, (labelx, keypointsx) in enumerate(loader):
        print("label = ", labelx)
        print("label size = ", labelx.size())
        for k in range(labelx.size()[0]):
            print("argmax = ", labelx[k].argmax().numpy().tolist())
        print("keypoints = ", keypointsx)
        print("size = ", keypointsx.size())
    print("样本数 = ", len(loader))
    print(loader.__len__())
