import json
import os
from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
import numpy as np

def print_json_crowd(root_path):
    with open(os.path.join(root_path, "livecell_coco_train_.json")) as fp:
        root = json.load(fp)
    print('Length of annotations:', len(root['annotations']))
    a = 0
    for ann in root['annotations']:
        if not ann["iscrowd"]:
            a+=1
    print(a)


coco_root = 'D:/workspace/dataset/Public/livecell/annotations/LIVECell'
cell_classes = ['a172','bt474','bv2','huh7','mcf7','shsy5y','skbr3','skov3']
# print_json_crowd(coco_root, )
annotation_file = os.path.join(coco_root, "livecell_coco_val_.json")

save_iscrowd = False

coco = COCO(annotation_file)
imgIds = coco.getImgIds()       # 图像ID列表
print("imgIds len: {}".format(len(imgIds)))

img_cnt = 0
crowd_cnt = 0

for idx, imgId in tqdm.tqdm(enumerate(imgIds), ncols=100):
    if save_iscrowd:
        annIds = coco.getAnnIds(imgIds=imgId)      # 获取该图像上所有的注释id->list
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)  # 获取该图像的iscrowd==0的注释id
    if len(annIds) > 0:
        image = coco.loadImgs([imgId])[0]
        ## ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']

        h, w = image['height'], image['width']
        gt_name = image['file_name'].replace('.tif', '.tiff')
        gt = np.zeros((h, w), dtype=np.uint8)
        anns = coco.loadAnns(annIds)    # 获取所有注释信息

        has_crowd_flag = 0
        save_flag = 0
        for ann_idx, ann in enumerate(anns):
            if not ann['iscrowd']:  # iscrowd==0
                segs = ann['segmentation']
                for seg in segs:
                    seg = np.array(seg).reshape(-1, 2)     # [n_points, 2]
                    cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], 1)
            else:
                has_crowd_flag = 1

        save_path = os.path.join(coco_root, gt_name)
        cv2.imwrite(save_path, gt)
        img_cnt += 1
        if has_crowd_flag:
            crowd_cnt += 1

        if idx % 100 == 0:
            print('Processed {}/{} images.'.format(idx, len(imgIds)))

print('crowd/all = {}/{}'.format(crowd_cnt, img_cnt))
