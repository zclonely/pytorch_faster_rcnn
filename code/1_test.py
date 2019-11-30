import json
import os
import cfg
import numpy as np

from PIL import Image


json_dir = 'F:\Zchao\Gj_anno\dataset\json'
img_dir = 'F:\Zchao\Gj_anno\dataset\img'

json_files = os.listdir(json_dir)
json_path = os.path.join(json_dir, '7d8be8b9afd0d4436c2146e8b528c0b9-asset' + '.json')
with open(json_path, 'r') as f:
    json_dic = json.load(f)
regions = json_dic['regions']
print(regions)
# img_name= json_dic['asset']['name']
# print(json_dic['asset']['name'])
# img_path = os.path.join(img_dir, json_dic['asset']['name'] + '.jpg')
# img =  Image.open(img_path)
# img = img.convert('RGB')
# img = np.array(img)