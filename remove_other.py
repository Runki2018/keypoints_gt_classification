
if __name__ == '__main__':
    # 去除数据集中的 “0-其他”类，该类的样本数过多，且类内相似度低，导致模型最终把所有样本都归结为该类。
    file = './combine_sample/test_annotations.txt'
    file1 = './combine_sample/test_annotations8.txt'
    with open(file, 'r') as f:
        with open(file1, 'w') as f1:
            for line in f.readlines():
                if line.split(' ')[0] != '0':
                    f1.write(line)
