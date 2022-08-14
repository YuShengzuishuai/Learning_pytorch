# -*- coding: utf-8 -*-
from math import log10
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, SwinModel, ConvNextModel, AutoFeatureExtractor
from transformers import BertTokenizer, BertModel
from PIL import Image
import cv2


class FakeEffectModel(nn.Module):
    def __init__(self, pretrain_name, mode, pretrain_type, is_train=True) -> None:
        super().__init__()
        if pretrain_type == 'vit':
            self.model = ViTModel.from_pretrained(pretrain_name)
        elif pretrain_type == 'swin':
            self.model = SwinModel.from_pretrained(pretrain_name)
        elif pretrain_type == 'conv':
            self.model = ConvNextModel.from_pretrained(pretrain_name)
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.fc1 = nn.Linear(768 * 2, 256)
        self.fc2 = nn.Linear(256, 2)
        self.mode = mode
        self.is_train = is_train
        self.softmax = torch.nn.Softmax()

    def forward(self, input_imgs, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state_text = bert_output['last_hidden_state']
        pooler_output_text = torch.mean(hidden_state_text, dim=1)
        batch_size, num_frames, num_channel, img_height, img_width = input_imgs.shape
        input_imgs = torch.reshape(input_imgs, (batch_size * num_frames, num_channel, img_height, img_width))
        model_output = self.model(pixel_values=input_imgs)
        hidden_state_imgs, pooler_output_imgs = model_output['last_hidden_state'], model_output['pooler_output']
        hidden_size_imgs = hidden_state_imgs.shape[2]

        pooler_output_imgs = torch.reshape(pooler_output_imgs, (batch_size, -1, hidden_size_imgs))
        pooler_output_imgs = torch.mean(pooler_output_imgs, dim=1, keepdim=False)
        pooler_output = torch.cat((pooler_output_imgs, pooler_output_text), dim=1)
        x = F.relu(self.fc1(pooler_output))
        logits = self.fc2(x)
        if self.is_train:
            return logits
        else:
            probs = self.softmax(logits)
            return probs


if __name__ == "__main__":
    import cv2

    pretrain_name = 'microsoft/swin-tiny-patch4-window7-224'
    mode = 'mean'
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = FakeEffectModel(pretrain_name=pretrain_name, pretrain_type='swin', mode=mode)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    roberta = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # input_text = tokenizer("字节跳动", return_tensors="pt")
    input_text = tokenizer("字节跳动", max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    # outputs = roberta(**input_text)
    input_images = torch.rand(1, 8, 3, 224, 224)
    print(torch.tensor(input_text['input_ids']).shape)
    # res = model(input_images, input_text['input_ids'], input_text['attention_mask'], input_text['token_type_ids'])
    # print(res.shape)

    # input_ids.shape:torch.Size([16, 1, 256]),input_mask.shape:torch.Size([16, 1, 256]),segment.shape:torch.Size([16, 1, 256]),imgs.shape:torch.Size([16, 8, 3, 224, 224])