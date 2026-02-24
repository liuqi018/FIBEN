r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetDeepglobe(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'deepglobe'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'Deepglobe', '04_train_cat')

        self.categories = ['1','2','3','4','5','6']

        self.class_ids = range(0, 6)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        # --- 保留原始图用于可视化 ---
        query_img_orig = np.array(query_img).astype(np.float32) / 255.0  # [H, W, C], 0~1
        support_imgs_orig = [np.array(img).astype(np.float32) / 255.0 for img in support_imgs]

        # --- 送入模型的 transform ---
        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(),
                                   query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(),
                                  support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        return (support_imgs, support_masks, query_img, query_mask, class_sample,
                support_names, query_name, support_imgs_orig, query_img_orig)

    def load_frame(self, query_name, support_names):
        # 读取原图
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        # 获取类别文件夹名
        parts = query_name.replace('\\', '/').split('/')
        class_id = parts[-4]  # 倒数第4层就是类别文件夹

        # groundtruth 文件夹路径
        ann_path = os.path.join(self.base_path, class_id, 'test', 'groundtruth')

        # query mask
        query_basename = os.path.splitext(os.path.basename(query_name))[0]  # 去掉后缀
        # 根据实际命名规则拼接
        # 705728.jpg -> 705728_mask_51.png
        query_mask_path = os.path.join(ann_path,
                                       query_basename.split('_')[0] + '_mask_' + query_basename.split('_')[-1] + '.png')

        # support masks
        support_mask_paths = []
        for s in support_names:
            s_basename = os.path.splitext(os.path.basename(s))[0]
            s_mask_path = os.path.join(ann_path,
                                       s_basename.split('_')[0] + '_mask_' + s_basename.split('_')[-1] + '.png')
            support_mask_paths.append(s_mask_path)

        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(p) for p in support_mask_paths]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id


    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat, 'test', 'origin'))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
