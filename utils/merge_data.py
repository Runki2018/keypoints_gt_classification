import json


def mergeData():
    """纠正后的错分类样本json文件，替换掉原json文件里相应的样本"""
    total_file = "../data/revise_totalSamples.json"
    revise_file = "../data/2021-04-16.json"  # misclassifiedSample.json"
    index_file = "../data/misclassified_index2021-04-16.txt"  # misclassifiedSample_index.txt
    save_file = "../data/revise_totalSamples2.json"
    # 读入文件
    total_samples = json.load(open(total_file, "r"))
    revise_samples = json.load((open(revise_file, "r")))
    f = open(index_file, "r")
    # 读入序号
    indexes = f.read()
    indexes = indexes.split("\n")  # [“（index, predict）”, ...]
    if indexes[-1] == "":
        indexes.pop()  # 去除最后一个空行
    f.close()

    images_list_total = total_samples["images"]
    annotations_list_total = total_samples["annotations"]
    images_list_revise = revise_samples["images"]
    annotations_list_revise = revise_samples["annotations"]
    for i, line in enumerate(indexes):
        index = line.strip("()").split(",")  # ["index", "predict"]
        index = int(index[0])
        images_list_total[index] = images_list_revise[i]
        annotations_list_total[index] = annotations_list_revise[i]

    total_samples["images"] = images_list_total
    total_samples["annotations"] = annotations_list_total
    json.dump(total_samples, open(save_file, "w"), indent=4)


if __name__ == '__main__':
    mergeData()
