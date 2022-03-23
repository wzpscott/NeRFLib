import cv2
import os
import os.path as osp
import shutil

data_dir = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego'
imgs_dir = osp.join(data_dir, 'train')
new_data_dir = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/toy_lego'
new_imgs_dir = osp.join(new_data_dir, 'train')

os.makedirs(new_data_dir, exist_ok=True)
os.makedirs(new_imgs_dir, exist_ok=True)

for img_dir in os.listdir(imgs_dir):
    img = cv2.imread(osp.join(imgs_dir, img_dir),cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (80,80))
    cv2.imwrite(osp.join(new_imgs_dir, img_dir), img)

shutil.copy(osp.join(data_dir, 'transforms_train.json'), osp.join(new_data_dir, 'transforms_train.json'))