import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import concurrent_process
from PIL import Image
from torch.utils.data import DataLoader
import cv2
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

cls_token = "[CLS]"
sep_token = "[SEP]"
pad_token_id = 0


class FakeEffectDataset(Dataset):
    def __init__(self, data_path, imgs_dir) -> None:
        super().__init__()
        self.data_dict = self.load_data(data_path)
        self.keys = list(self.data_dict.keys())
        self.imgs_dir = imgs_dir
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.max_len = 256

    def load_data(self, data_path):
        fp = open(data_path, 'r')
        data_set = json.load(fp)
        return data_set

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        video_id = self.keys[index]
        sample = self.data_dict[video_id]
        product_name = sample['product_name'] if type(sample['product_name']) == str else ''
        first_name_new = sample['first_name_new'] if type(sample['first_name_new']) == str else ''
        second_name_new = sample['second_name_new'] if type(sample['second_name_new']) == str else ''
        third_name_new = sample['third_name_new'] if type(sample['third_name_new']) == str else ''
        valid_text = product_name + '#' + first_name_new + '#' + second_name_new + '#' + third_name_new
        input_ids, input_mask, segment_ids = self.text_transform(valid_text)
        label = torch.tensor(int(sample['label']))
        input_imgs = self.img_transform(video_id)
        return input_ids, input_mask, segment_ids, input_imgs, label

    def img_transform(self, video_id):
        pic_paths = [os.path.join(self.imgs_dir, video_id + '_' + str(i) + '.jpg') for i in range(8)]
        img_data_list = concurrent_process(self.cv2_check, pic_paths, len(pic_paths))
        img_data_sample = np.stack(img_data_list, axis=0)
        img_data_sample = img_data_sample[:, :, :, [2, 1, 0]]  # BGR --> RGB
        img_data_sample = img_data_sample.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
        img_data_sample = (img_data_sample - mean) / std
        return np.transpose(img_data_sample, axes=(0, 3, 1, 2))

    def cv2_check(self, pic_paths):
        img = cv2.imread(pic_paths)
        img = cv2.resize(img, (224, 224))
        return img

    def text_transform(self, text):
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[: (self.max_len - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0, ] * len(tokens)
        input_mask = [1, ] * len(tokens)

        # padding
        padding_length = self.max_len - len(tokens)
        input_ids += [pad_token_id] * padding_length
        input_mask += [0, ] * padding_length
        segment_ids += [0, ] * padding_length

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)


if __name__ == '__main__':
    data_path = '/mnt/bd/cv-data/data/fake_effect/new_eval_with_text.json'
    img_path = '/mnt/bd/cv-data/image'
    my_dataset = FakeEffectDataset(data_path, img_path)

    train_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=False, num_workers=64)
    st = time.time()
    for idx, (input_ids, input_mask, segment_ids, input_imgs, label) in tqdm(enumerate(train_dataloader)):
        print(input_imgs.shape, input_ids.shape)
        # print(input_imgs_batch.shape,label_batch.shape)
        break