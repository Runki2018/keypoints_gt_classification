import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, samples, transform=None):
        self.label_list = samples[0]  # list
        self.keypoints_list = samples[1]  # list
        self.transform = transform
        self.n_class = 8

    def __len__(self):
        """数据集样本数"""
        return len(self.label_list)

    def __getitem__(self, idx):
        """每次返回一个样本的标签和关键点"""
        keypoints = torch.tensor(self.keypoints_list[idx])
        # label = torch.zeros((1, 8), dtype=torch.float)  # one-hot code for MSELoss
        # label[0][self.label_list[idx]-1] = 1  # one-hot code for MSELoss
        label = torch.tensor(self.label_list[idx], dtype=torch.long) - 1
        return label, keypoints


class MyDataLoader:
    """数据加载器，用于加载自定义的数据集"""

    def __init__(self, batch_size=19):
        self.train_list = read_txt('./combine_sample/train_annotations8.txt')
        self.test_list = read_txt('./combine_sample/test_annotations8.txt')
        self.BATCH_SIZE = batch_size  # 一次读入多少个样本
        self.num_workers = 2  # 加载batch的线程数

    def train(self):
        training_set = MyDataSet(samples=self.train_list,
                                 transform=transforms.ToTensor())
        train_loader = DataLoader(
            dataset=training_set, batch_size=self.BATCH_SIZE,
            shuffle=True, num_workers=self.num_workers)
        return train_loader

    def test(self):
        training_set = MyDataSet(samples=self.test_list,
                                 transform=transforms.ToTensor())
        test_loader = DataLoader(
            dataset=training_set, batch_size=self.BATCH_SIZE,
            shuffle=True, num_workers=self.num_workers)
        return test_loader


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
    count = [0 for _ in range(8)]
    for x in label_list:
        count[x-1] += 1
    for i in range(8):
        print("类别 {}\t的数量为 {} ".format(classes[i + 1], count[i]))
    return label_list, keypoints_list


if __name__ == '__main__':
    b = 20
    loader = MyDataLoader(batch_size=b).test()
    for i, (label, keypoints) in enumerate(loader):
        print("label = ", label)
        print("label size = ", label.size())
        for k in range(label.size()[0]):
            print("argmax = ", label[k].argmax().numpy().tolist())
        print("keypoints = ", keypoints)
        print("size = ", keypoints.size())
    print("样本数 = ", len(loader))
    print(loader.__len__())
