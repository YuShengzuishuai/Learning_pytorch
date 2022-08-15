from datasets import load_dataset
import numpy as np
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import torch
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from tqdm import tqdm
import torchmetrics
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import argparse  # 必须引入 argparse 包
import random

def load_data(train_data_dir,validation_data_dir,opt):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    cache_dir = '/opt/tiger/hard_video/cache'
    dataset = load_dataset('csv', data_files={'train': train_data_dir,
                                                'validation':validation_data_dir},cache_dir=cache_dir)

    def tokenize_function(examples):
        examples['text'] = [t if t else "" for t in examples['text']]
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function,batched=True, num_proc=20,load_from_cache_file=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_sampler = DistributedSampler(tokenized_datasets['train'])
    train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], sampler=train_sampler, batch_size=opt.batch_size)  # 注意这里的batch_size是每个GPU上的batch_size
    val_sampler = DistributedSampler(tokenized_datasets['validation'])
    val_loader = torch.utils.data.DataLoader(tokenized_datasets['validation'], sampler=val_sampler, batch_size=opt.batch_size*4)  # 注意这里的batch_size是每个GPU上的batch_size
    return train_loader,val_loader

def compute_metrics(logits, labels,epoch):
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions,average='micro')
    acc = accuracy_score(labels, predictions)
    print ('Eval Metrics {}: accuracy:{},f1:{},pre:{},recall:{}'.format(epoch,acc,f1,precision,recall),flush=True)

def train(train_loader,val_loader,args,device):
    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()
    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext",num_labels=2)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_training_steps = args.epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(args.epochs):
        batch_idx = 0
        model.train()
        for batch in train_loader:
            # batch_idx+=1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            if batch_idx % 100 == 0 and args.local_rank==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx, len(train_loader), loss.item()))
                print(len(train_loader),loss.item())
            batch_idx+=1

        if args.local_rank==0:
            torch.save(model.state_dict(),'/opt/tiger/hard_video/model/model_{}.pth'.format(epoch))
            print('model save /opt/tiger/hard_video/model/model_{}.pth'.format(epoch))
            os.popen('hdfs dfs -put -f /opt/tiger/hard_video/model/ /user/yusheng/data/hard_video/').read()
            print('model save success to hdfs /opt/tiger/hard_video/model/model_{}.pth'.format(epoch))


        model.eval()
        labels_list = []
        logits_list = []
        for batch in tqdm(val_loader):
            labels_list.append(batch['labels'])
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predict_scores = F.softmax(outputs.logits, dim=1)
                logits_list.append(predict_scores.cpu().detach().numpy())
        labels_list = np.concatenate(labels_list)
        logits_list = np.concatenate(logits_list)
        compute_metrics(logits_list, labels_list, epoch)

def main(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data_dir = '/opt/tiger/hard_video/train.csv'
    validation_data_dir = '/opt/tiger/hard_video/val.csv'
    train_dataloader,val_dataloader = load_data(train_data_dir,validation_data_dir,args)
    train(train_dataloader,val_dataloader,args,device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, default=10)
    parser.add_argument('--num_classes',type=int,default=2)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    opt = parser.parse_args()

    data_dir = '/opt/tiger/hard_video/model/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.popen('hdfs dfs -get /user/yusheng/data/hard_video/train.csv /opt/tiger/hard_video/train.csv').read()
    os.popen('hdfs dfs -get /user/yusheng/data/hard_video/val.csv /opt/tiger/hard_video/val.csv').read()
    accuracy = torchmetrics.Accuracy(num_classes=2)
    main(opt)
    # os.popen('hdfs dfs -put -f /opt/tiger/hard_video/model/ /user/yusheng/data/hard_video/').read()