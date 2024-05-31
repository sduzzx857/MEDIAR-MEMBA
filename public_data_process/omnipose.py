import shutil
import glob
import os

root = '/home/data/MEDIAR/Public/omnipose/bact_phase/train_sorted'
img_save = '/home/data/MEDIAR/Public/images'
lab_save = '/home/data/MEDIAR/Public/labels'
classes = ['5I_crop', 'A22', 'bthai', 'caulo', 'cex', 'dnaA', 'ecoli_mut', 'francisella', 'ftsN', 'hpylori', 'murA', 'PAO1_staph', 'streptomyces', 'serratia', 'PSVB', 'vibrio', 'wiggins'] # ['A22', 'bthai', 'cex', 'vibrio', 'wiggins']

for c in classes:
    flow_list = glob.glob(os.path.join(root, c, "*flows.tif"))
    mask_list = glob.glob(os.path.join(root, c, "*masks.tif"))
    all_list = glob.glob(os.path.join(root, c, "*.tif"))
    img_list = [i for i in all_list if i not in (flow_list+mask_list)]
    print(len(flow_list))
    print(len(mask_list))
    print(len(all_list))
    print(len(img_list))
    for img in img_list:
        name = os.path.split(img)[-1]
        shutil.copy(img, os.path.join(img_save, name))
    for mask in mask_list:
        name = os.path.split(mask)[-1].replace("tif", "tiff")
        shutil.copy(mask, os.path.join(lab_save, name))
