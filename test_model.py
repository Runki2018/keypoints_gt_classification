import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import classifier
from mydataset import MyDataLoader
from utils.parse_config import parse_model_cfg


# from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

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
        print("类别 {}\t的数量为 {} / {}".format(classes[idx], sum_true[idx], sum_gt[idx]), end=', ')
        print("precision = {:.4}, recall = {:.4}".format(precision[idx], recall[idx]))
    ap = precision.sum() / sum_gt.size
    ar = recall.sum() / sum_gt.size
    print("AP = {}, AR = {}".format(ap, ar))


if __name__ == '__main__':
    cfg_path = "./cfg/network.cfg"
    pt_file = "./runs/2021-03-29/94acc_109epoch_128_64.pt"
    net_block = parse_model_cfg(cfg_path)[0]  # [net]
    n_classes = net_block["n_classes"]
    batch_size = net_block["batch"]

    classes = ['0-other', '1-OK', '2-palm', '3-up', '4-down', '5-right', '6-left', '7-heart', '8-hush']
    if n_classes == 8:
        classes.pop(0)

    model = classifier(cfg_path).cuda()
    test_loader = MyDataLoader(batch_size=batch_size).train()
    param_dict = torch.load(pt_file)
    model.load_state_dict(param_dict["model_state"])

    model.eval()
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for label, keypoints in tqdm(test_loader):
            y_predict = model(keypoints.cuda(0))
            predict_index = y_predict.argmax(dim=1, keepdim=False)  # tensor([y0, y1, ..., y_batch])
            label = label.cuda() if torch.cuda.is_available() else label
            for y_true, y_pred in zip(label, predict_index):
                confusion_matrix[y_true, y_pred] += 1  # 获取混淆矩阵

    # calculate Precision and Recall
    metric_eval(confusion_matrix)
    # draw the confusion_matrix
    plot_confusion_matrix(confusion_matrix, 'confusion_matrix.png', title='confusion matrix')
