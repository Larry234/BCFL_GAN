import ast
import web3
import time
import torch
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_model(client, hs, ToTensor=True):

    # returns global model state dict
    dict_str = client.cat(hs).decode("UTF-8")
    global_model = ast.literal_eval(dict_str)
    if ToTensor:
        res = {k: torch.FloatTensor(v) for k, v in global_model.items()}
    return res

def merge_dict(x,y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v

def FedAvg(hs):

    nums = 0
    global_res_dict = dict()
    for h in hs:
        nums += 1
        data = load_model(h)
        merge_dict(data, global_res_dict)

    for k, v in global_res_dict.items():
        global_res_dict[k] = v / nums
        global_res_dict[k] = global_res_dict.tolist()
    
    return global_res_dict




