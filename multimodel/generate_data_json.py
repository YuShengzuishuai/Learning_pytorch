import pandas as pd
from sklearn.model_selection import  train_test_split
import json
data = pd.read_csv('/mnt/bd/cv-data/train_data.csv',usecols=['object_id','ocr_text','label','hit_invalid_img'],nrows=1000)
# data = data[data['hit_invalid_img']!=1]
print(data[data['hit_invalid_img']==1].object_id.tolist()[:5])
data.reset_index(drop=True,inplace=True)
print('data.shape:',data.shape)
train_set,eval_set = train_test_split(data,test_size=0.05)

print('trian_set.shape:',train_set.shape)
print('eval_set.shape:',eval_set.shape)
train_dict = {}
for i,row in train_set.iterrows():
    train_dict[row['object_id']] = {
        'ocr':row['ocr_text'],
        'label':row['label']
    }

eval_dict = {}
for i,row in eval_set.iterrows():
     eval_dict[row['object_id']] = {
       'ocr':row['ocr_text'],
        'label':row['label']
    }

train_json = open('/mnt/bd/cv-data/train_dict.json','w')
json.dump(train_dict,train_json)
train_json.close()


eval_json = open('/mnt/bd/cv-data/eval_dict.json','w')
json.dump(eval_dict,eval_json)
eval_json.close()