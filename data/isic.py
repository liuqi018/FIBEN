r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetISIC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'isic'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'ISIC')
        self.categories = ['1','2','3']

        self.class_ids = range(0, 3)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        # --------- 保留原始 RGB 图像（numpy, 0~1） ---------
        orig_query_img = np.array(query_img).astype(np.float32) / 255.0
        orig_support_imgs = [np.array(s).astype(np.float32) / 255.0 for s in support_imgs]

        # --------- 处理后的 Tensor ---------
        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:],
            mode='nearest'
        ).squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode='nearest'
            ).squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        # --------- 返回 Tensor + 原始图像 ---------
        return support_imgs, support_masks, query_img, query_mask, class_sample, support_names, query_name, orig_support_imgs, orig_query_img

    def load_frame(self, query_name, support_names):
        # 读取原图
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        # GroundTruth 文件夹
        ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')

        # query mask 路径
        query_basename = os.path.splitext(os.path.basename(query_name))[0]
        query_mask_path = os.path.join(ann_path, query_basename + '_segmentation.png')

        # support mask 路径
        support_mask_paths = [
            os.path.join(ann_path, os.path.splitext(os.path.basename(name))[0] + '_segmentation.png')
            for name in support_names
        ]

        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(path) for path in support_mask_paths]

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

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
