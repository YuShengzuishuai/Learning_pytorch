# coding: utf-8
# this file will be executed by main.sh

import os
import cv2
import random
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
unique_id_set = set()
def download_sample(unique_id, sample_urls):
    for idx, url in enumerate(sample_urls):
        save_path = '/mnt/bd/cv-data/image_test/' + str(unique_id) + '_' + str(idx) + '.jpg'
        count = 0
        while True:
            try:
                count += 1
                if os.path.exists(save_path):
                    break
                img_data = requests.get(url, timeout=10, allow_redirects=True).content
                img = np.asarray(bytearray(img_data), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                cv2.imwrite(save_path, img)
                break
            except Exception as e:
                if count >= 3:
                    unique_id_set.add(unique_id)
                    break
        if count >= 3:
            return
    return
def concurrent_process_xy(func, input_xs, input_ys, num_threads):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for res in t.map(func, input_xs, input_ys):
            results.append(res)
    return results

data = pd.read_csv('***')
data = data.fillna('')
video_id_list = data.object_id.tolist()
data['sample_urls'] = data.sample_urls.progress_map(lambda x: eval(x))
urls_list = data.sample_urls.tolist()



# labels = data.label.tolist()
# ocr_text_list = data.ocr_text.tolist()

import time
st = time.time()
for start in tqdm(range(0, len(video_id_list), 500)):
    if start % 500 == 0:
        print("processing: ",  start, "cost time is:", time.time() - st)
    end = min(start + 500, len(video_id_list))
    video_batch = video_id_list[start:end]
    urls_batch = urls_list[start:end]
    concurrent_process_xy(download_sample, video_batch, urls_batch, len(video_batch))

from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')
data['hit_invalid_img'] = data.object_id.progress_map(lambda x : 1 if x in list(unique_id_set) else 0)
data.to_csv('/mnt/bd/cv-data/fake_effect_test_data.csv',index = 0)