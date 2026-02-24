r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

import pathlib


class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        project_root = pathlib.Path(__file__).parent.parent.resolve()  # IFA-master
        self.base_path = os.path.join(project_root, datapath, 'LungSegmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.base_path = os.path.normpath(self.base_path)
        self.img_path = os.path.normpath(self.img_path)
        self.ann_path = os.path.normpath(self.ann_path)

        self.categories = ['1']
        self.class_ids = range(0, 1)
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

        # --------- Tensor 处理 ---------
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
        # 读取 mask
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        # 原图路径对应 mask
        query_img_file = os.path.basename(query_name).replace('_mask', '')
        query_img = Image.open(os.path.join(self.img_path, query_img_file)).convert('RGB')

        support_imgs = []
        for s_mask in support_names:
            s_img_file = os.path.basename(s_mask).replace('_mask', '')
            s_img_path = os.path.join(self.img_path, s_img_file)
            support_imgs.append(Image.open(s_img_path).convert('RGB'))

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
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            folder = self.ann_path
            folder = os.path.normpath(folder)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist!")
                continue
            files = sorted(os.listdir(folder))
            for f in files:
                if os.path.splitext(f)[1].lower() == '.png':
                    img_metadata_classwise[cat].append(os.path.join(folder, f))

        return img_metadata_classwise
