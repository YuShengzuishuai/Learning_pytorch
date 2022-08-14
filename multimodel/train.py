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


def reduce_value(value, average=True):
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= 6
        return value


def train(model, train_loader, eval_loader, device, args, eval_size, optimizer):
    num_training_steps = args.epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        mean_loss = torch.zeros(1).to(device)
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, input_imgs, label_batch = batch[0], batch[1], batch[2], batch[3], batch[
                4]
            logits = model(input_imgs, input_ids, input_mask, segment_ids)
            loss = loss_fn(logits, label_batch)
            loss.backward()
            loss = reduce_value(loss, average=True)

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if step % 100 == 0 and args.local_rank == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, step, len(train_loader),
                                                                 round(mean_loss.item(), 3)))

        if args.local_rank == 0:
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), '/mnt/bd/cv-data/multimodal/model_{}.pth'.format(epoch))
            print('model save /mnt/bd/cv-data/multimodal/model_{}.pth'.format(epoch))
            # os.popen('hdfs dfs -put -f /opt/tiger/hard_video/model/ /user/yusheng/data/hard_video/').read()
            # print('model save success to hdfs /opt/tiger/hard_video/model/model_{}.pth'.format(epoch))

        model.eval()
        sum_num = torch.zeros(1).to(device)
        sum_loss = torch.zeros(1).to(device)
        for step, batch in enumerate(eval_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, input_imgs, label_batch = batch[0], batch[1], batch[2], batch[3], batch[
                4]
            with torch.no_grad():
                logits = model(input_imgs, input_ids, input_mask, segment_ids)
                loss = loss_fn(logits, label_batch)
                sum_loss += loss
                predict_scores = F.softmax(logits, dim=1)
                pred = torch.max(predict_scores, dim=1)[1]
                sum_num += torch.eq(pred, label_batch).sum()
        sum_num = reduce_value(sum_num, average=False)
        sum_loss = reduce_value(sum_loss, average=True)
        if args.local_rank == 0:
            print('Eval acc:{},loss:{}'.format(sum_num.item() / eval_size, sum_loss.item()))


def main(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data_path = '/mnt/bd/cv-data/data/fake_effect/new_train_with_text.json'
    img_path = '/mnt/bd/cv-data/image'
    train_dataset = FakeEffectDataset(train_data_path, img_path)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                               batch_size=opt.batch_size)  # 注意这里的batch_size是每个GPU上的batch_size

    eval_data_path = '/mnt/bd/cv-data/data/fake_effect/new_eval_with_text.json'
    eval_dataset = FakeEffectDataset(eval_data_path, img_path)
    eval_sampler = DistributedSampler(eval_dataset)
    eval_size = eval_sampler.total_size
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=opt.batch_size * 4, shuffle=False,
                                              sampler=eval_sampler)  # 注意这里的batch_size是每个GPU上的batch_size

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    pretrain_name = 'microsoft/swin-tiny-patch4-window7-224'
    # pretrain_name = 'facebook/convnext-tiny-224'
    # pretrain_name = 'google/vit-base-patch16-224-in21k'
    mode = 'mean'
    model = FakeEffectModel(pretrain_name=pretrain_name, pretrain_type='swin', mode=mode)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    train(model, train_loader, eval_loader, device, args, eval_size, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=6, type=int,
                        help='number of distributed processes')
    opt = parser.parse_args()
    main(opt)
    # os.popen('hdfs dfs -put -f /opt/tiger/hard_video/model/ /user/yusheng/data/hard_video/').read()