import torch
import time
import os
from torch import nn
from mydataset import MyDataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from FC_module import MyFC
from model import classifier
from utils.parse_config import parse_model_cfg


class runModel:
    def __init__(self):
        super(runModel, self).__init__()
        self.keep_going = False
        self.load_path = "weight/91acc_9category_1024_512_256.pt"  # 继续训练时，要加载的参数文件
        self.batch_size = 30  # 分批训练数据、每批数据量
        self.learning_rate = 0.01  # 1e-2  # 学习率
        self.num_epoches = 700 # 训练次数
        cfg_path = "./cfg/network.cfg"
        net_block = parse_model_cfg(cfg_path)[0]  # [net]
        self.n_classes = net_block["n_classes"]  # 类别数
        # 保存用于可视化观察参数变化的列表
        self.loss_list, self.lr_list, self.acc_list = [], [], []  # 记录损失值\学习率\准确率变化
        self.layer_list = [1024, 512, 256]
        # self.model = MyFC(self.layer_list, 42, self.n_classes)
        self.model = classifier("./cfg/network.cfg")
        self.train_loader = MyDataLoader(batch_size=self.batch_size).train()
        self.test_loader = MyDataLoader(batch_size=self.batch_size).test()
        self.best_acc = 0  # 最高准确类
        self.writer = SummaryWriter(logdir='./log')  # 记录训练日志

    def train_model(self):
        self.model.train()  # 训练模式
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model = self.model.cuda(0)
        else:
            device = torch.device('cpu')

        criterion = nn.CrossEntropyLoss().cuda(0)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # criterion = nn.MSELoss().cuda(0)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        start_epoch = 0
        if self.keep_going:  # 继续
            param_dict = self.load_model(self.load_path)
            self.model.load_state_dict(param_dict["model_state"])
            optimizer.load_state_dict(param_dict["optimizer_state"])
            start_epoch, self.lr_list = param_dict["epoch"], param_dict["lr_list"]
            self.loss_list, self.acc_list = param_dict["loss_list"], param_dict["acc_list"]

        # train:
        for epoch in range(start_epoch, self.num_epoches):
            print('epoch :', epoch)
            print('*' * 10)
            loss = 0
            sum_loss = 0
            for i, (label, keypoints) in enumerate(self.train_loader, 1):
                optimizer.zero_grad()
                # label = label.squeeze().cuda(0)  # for MSELoss
                label = label.cuda(0)
                y_pred = self.model(keypoints.cuda(0))
                loss = criterion(y_pred, label)
                # 反向传播：
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    sum_loss += loss.item()
                    # print(self.model.state_dict())
            scheduler.step(epoch)
            # 更新变化列表，记录loss和lr变化
            with torch.no_grad():
                self.loss_list.append(sum_loss)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                self.lr_list.append(lr)
                acc = self.test_model()
                self.writer.add_scalar('loss', loss.item(), epoch)
                self.writer.add_scalar('sum_loss', sum_loss, epoch)
                self.writer.add_scalar('lr', lr, epoch)
                self.writer.add_scalar('accuracy', acc, epoch)

            if acc > self.best_acc or epoch == self.num_epoches - 1:
                self.best_acc = acc
                # 存放信息的字典:
                param_dict = {'model_state': self.model.state_dict(),
                              'loss': loss,
                              'optimizer_state': optimizer.state_dict(),
                              'epoch': epoch,
                              'loss_list': self.loss_list,
                              'lr_list': self.lr_list,
                              'acc_list': self.acc_list}
                self.save_model(model_param=param_dict, acc=acc, epoch=epoch)

    def test_model(self):
        self.model.eval()
        n_classes = self.n_classes  # 类别数
        classes = ['0-其他', '1-OK', '2-手掌', '3-向上', '4-向下', '5-向右', '6-向左', '7-比心', '8-嘘']
        sum_true = [0 for _ in range(n_classes)]  # 正确个数
        with torch.no_grad():
            for label, keypoints in tqdm(self.test_loader):
                y_predict = self.model(keypoints.cuda(0))
                predict_index = y_predict.argmax(dim=1, keepdim=False)  # tensor([y0, y1, ..., y_batch])
                # label = label.argmax(dim=1, keepdim=False)  # MSELoss
                label = label.cuda() if torch.cuda.is_available() else label
                for y, y_pred in zip(label, predict_index):
                    if y == y_pred:
                        sum_true[y] += 1
        print("label = \t", label)
        print("predict_index = \t", predict_index)
        for i in range(n_classes):
            print("类别 {}\t的数量为 {} ".format(classes[i], sum_true[i]))  # 8类
        total = len(self.test_loader) * self.batch_size
        accuracy = sum(sum_true) / total
        print("sum_true = {}, len = {} , total = {} ".format(sum_true, len(self.test_loader), total))
        print("accuracy = ", accuracy)
        self.acc_list.append(accuracy)
        return accuracy

    def save_model(self, model_param, acc=0., epoch=0):
        save_dir = './runs/' + time.strftime("%Y-%m-%d", time.localtime())
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_name = "/{}acc_{}epoch_{}category.pt".format(int(acc * 100), epoch,
                                                          self.n_classes)
        save_file = save_dir + file_name
        torch.save(model_param, save_file)

    @staticmethod
    def load_model(load_path):
        return torch.load(load_path)


if __name__ == '__main__':
    runModel().train_model()
    file = "./log/"
