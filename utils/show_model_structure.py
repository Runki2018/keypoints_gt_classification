import torch


def show_model_dict():
    path = "../runs/2021-03-30/95acc_8category_1024_512_256.pt"
    parm_dict = torch.load(path)
    state_dict = parm_dict["model_state"]
    keys = state_dict.keys()
    for key in keys:
        print(key, "\t", state_dict[key].shape)


if __name__ == '__main__':
    show_model_dict()
