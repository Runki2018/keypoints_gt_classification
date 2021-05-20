import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import classifier
from mydataset import MyDataLoader
from utils.parse_config import parse_model_cfg
import json


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)  # x列号矩阵， y行号矩阵
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c == 0:
            continue  # 样本个数为0的就不显示数字
        color = 'red' if x_val != y_val else 'green'  # 对角线元素为绿色，其余为红色
        if title != 'Confusion Matrix':
            plt.text(x_val, y_val, "%0.2f" % (c,), color=color, fontsize=15, va='center', ha='center')  # 显示小数位
        else:
            plt.text(x_val, y_val, "%d" % (c,), color=color, fontsize=15, va='center', ha='center')  # 整数

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


def metric_eval(cm):
    """通过传入cm混淆矩阵，计算P,R,AP,AR等参数"""
    sum_gt = cm.sum(1)  # 各类样本的真实个数 [n1, n2, ...]
    sum_predict = cm.sum(0)  # 各类样本的预测个数 [n1, n2, ...]
    sum_true = cm.diagonal()  # 对角线元素为预测正确的个数
    precision = sum_true / sum_predict  # 查准率
    recall = sum_true / sum_gt  # 查全率
    for idx in range(sum_gt.size):
        print("类别 {}\t的数量为 {} / {}".format(classes[idx], sum_true[idx], sum_gt[idx]), end=',')
        print("precision = {:.4}, recall = {:.4}".format(precision[idx], recall[idx]))
    ap = precision.sum() / sum_gt.size
    ar = recall.sum() / sum_gt.size
    print("AP = {}, AR = {}".format(ap, ar))


def getMisclassifiedSample(tuple_list):
    save_dir = "./data/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("idx = ", tuple_list)
    json_file = json.load(open("./combine_sample/total_checked_samples.json", "r"))
    images_list = json_file["images"]
    annotations_list = json_file["annotations"]
    images_list = [images_list[x] for x, _ in tuple_list]  # x, _ = index, y_pred
    for i, (_, y) in enumerate(tuple_list):
        images_list[i]['predict'] = y
    annotations_list = [annotations_list[x] for x, _ in tuple_list]
    json_file["images"], json_file["annotations"] = images_list, annotations_list
    save_file = save_dir + "misclassified" + time.strftime("%Y-%m-%d", time.localtime()) + ".json"
    json.dump(json_file, open(save_file, "w"), indent=4)
    save_file = save_dir + "misclassified_index" + time.strftime("%Y-%m-%d", time.localtime()) + ".txt"
    with open(save_file, "w") as f:
        for index in tuple_list:
            f.write(str(index) + "\n")


def sort_confidence(preds):
    """分析错分样本的置信度"""
    values, indexes = preds.topk(k=n_classes, dim=-1)  # 将置信度前三的从大到小排列得到v，每个置信度相应的标签为index
    # values, indexes = values.tolist(), indexes.tolist()
    # values = [round(v, 3) for v in values]
    return values, indexes


if __name__ == '__main__':
    cfg_path = "./cfg/network.cfg"
    pt_file = "runs/2021-03-31/91acc_9category_1024_512_256.pt"
    # pt_file = "runs/2021-04-16/93acc_699epoch_9category.pt"
    net_block = parse_model_cfg(cfg_path)[0]  # [net]
    n_classes = net_block["n_classes"]
    batch_size = net_block["batch"]
    model = classifier(cfg_path).cuda()
    param_dict = torch.load(pt_file)
    model.load_state_dict(param_dict["model_state"])
    data_loader = MyDataLoader(batch_size=batch_size).test()
    confidence_threshold = 0.98  # 置信度阈值，大于该值才预测为真，否则pass
    # torch.save(model, './classification_model.pth')
    model = torch.load('./classification_model.pth')

    classes = ['0-other', '1-OK', '2-palm', '3-up', '4-down', '5-right', '6-left', '7-heart', '8-hush']
    if n_classes == 8:
        classes.pop(0)

    model.eval()
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    confidence_matrix = np.zeros((n_classes, n_classes, n_classes), dtype=float)  # 置信度和
    sum_matrix = np.zeros((n_classes, n_classes, n_classes), dtype=int)  # 每个置信度和由多少个样本得来
    false_idx = []  # 记录错分样本的序号，即txt文本中的行号

    with torch.no_grad():
        for label, keypoints in tqdm(data_loader):
            y_predict = model(keypoints.cuda(0))  # (batch, n_class) 9个类别的置信度
            predict_index = y_predict.argmax(dim=1, keepdim=False)  # tensor([y0, y1, ..., y_batch]), (batch)
            label = label.cuda() if torch.cuda.is_available() else label  # [a1, a2, ..., a_batch], (batch)
            i = -1
            for y_true, y_pred in zip(label, predict_index):  # [p1, p2, ..., p_batch]
                i += 1
                if y_predict[i, y_pred] < confidence_threshold:
                    continue
                confusion_matrix[y_true, y_pred] += 1  # 获取混淆矩阵
            values, indexes = y_predict.topk(k=n_classes, dim=-1)  # 将置信度及其相应序号，按置信度由大到小排序
            batch_size = label.size()[0]
            for i in range(batch_size):
                for j in range(n_classes):
                    confidence_matrix[label[i], indexes[i, j], j] += values[i, j]
                    sum_matrix[label[i], indexes[i, j], j] += 1
        # confidence_matrix.png /= sum_matrix
        with np.errstate(divide='ignore', invalid='ignore'):  # 防止报错
            confidence_matrix /= sum_matrix
            confidence_matrix = np.nan_to_num(confidence_matrix)  # 将数组中 nan值设置为0, inf设置为有限的大数

        confidence_matrix = confidence_matrix.round(3)  # 保留3个小数位
    for i in range(n_classes):
        print("类别：", classes[i])
        print(confidence_matrix[i])
        print("-" * 50)

    print(confidence_matrix[..., 0])

    # with torch.no_grad():
    #     for idx, label, keypoints in tqdm(data_loader):
    #         y_predict = model(keypoints.cuda(0))
    #         predict_index = y_predict.argmax(dim=1, keepdim=False)  # tensor([y0, y1, ..., y_batch])
    #         label = label.cuda() if torch.cuda.is_available() else label
    #         for b, (i, y_true, y_pred) in enumerate(zip(idx, label, predict_index)):
    #             if y_predict[b, y_pred] < confidence_threshold:
    #                 continue
    #             confusion_matrix[y_true, y_pred] += 1  # 获取混淆矩阵
    #             if y_true != y_pred:
    #                 false_idx.append((i.tolist(), y_pred.tolist()))
    # getMisclassifiedSample(false_idx)  # get Misclassified sample

    # calculate Precision and Recall
    metric_eval(confusion_matrix)
    # draw the confusion_matrix
    plot_confusion_matrix(confusion_matrix, 'confusion_matrix.png', title='Confusion Matrix')  # 个数
    plot_confusion_matrix(confidence_matrix[..., 0], 'confidence_matrix.png',
                          title='Confidence Matrix')  # 概率




