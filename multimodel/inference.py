import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import argparse  # 必须引入 argparse 包
import random
from model import FakeEffectModel
import torch.distributed as dist
from dataset import FakeEffectDataset
import pandas as pd
import torch.nn as nn
from collections import OrderedDict


def predict(model, test_loader):
    prob_list = []
    video_list = []
    result = {}
    for step, batch in tqdm(enumerate(test_loader)):
        images, label_batch = batch
        with torch.no_grad():
            logits = model(images.cuda())
            softmax = torch.nn.Softmax()
            probs = softmax(logits).cpu().detach().numpy()
            probs = probs[:, -1]
        for i in zip(label_batch.tolist(), probs.tolist()):
            result[str(i[0])] = i[1]
    return result


def main(args):
    train_data_path = '/mnt/bd/cv-data/test_dict.json'
    img_path = '/mnt/bd/cv-data/image_test'
    test_dataset = FakeEffectDataset(train_data_path, img_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                              shuffle=False)  # 注意这里的batch_size是每个GPU上的batch_size
    data = pd.read_csv('fake_effect_test_data.csv')
    pretrain_name = 'microsoft/swin-tiny-patch4-window7-224'
    mode = 'mean'
    model = FakeEffectModel(pretrain_name=pretrain_name, pretrain_type='swin', mode=mode)
    model_dir = os.listdir('/mnt/bd/cv-data/model')
    count = 0
    for model_name in model_dir:
        count += 1
        loaded_dict = torch.load('/mnt/bd/cv-data/model/' + model_name)
        new_state_dict = OrderedDict()
        for k, v in loaded_dict.items():
            name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v  # 新字典的key值对应的value一一对应
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        result = predict(model, test_loader)
        result_name = 'model' + str(count)
        data[result_name] = data.object_id.map(lambda x: result.get(str(x), -1))
        print('data.result.value_shape:', data[data[result_name] != -1].shape)
        data.to_csv('fake_effect_test_data.csv', index=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=6, type=int,
                        help='number of distributed processes')
    opt = parser.parse_args()
    main(opt)
    # os.popen('hdfs dfs -put -f /opt/tiger/hard_video/model/ /user/yusheng/data/hard_video/').read()