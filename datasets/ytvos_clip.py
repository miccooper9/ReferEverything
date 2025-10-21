"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset

import datasets.transforms as T
#import transforms as T
import os
from PIL import Image
import json
import numpy as np
import random

#from datasets.categories import ytvos_category_dict as category_dict
import argparse



category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}


im_transform = T.Compose([
    T.Resize(512,512),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

im_transformcog = T.Compose([
    T.Resize(384,384),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

im_transformwan = T.Compose([
    T.Resize(480,832),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


im_transform256 = T.Compose([
    T.Resize(256,256),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])








class YTVOSDatasetVid(Dataset):


    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool,
                 num_frames: int, max_skip: int, sampler_interval: int, args=None):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.num_clips = args.num_clips
        self.max_skip = max_skip #not used
        self.sampler_interval = sampler_interval
        #self.reverse_aug = args.reverse_aug
        # create video meta data
        self.prepare_metas()

        print('\n Vid-- video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        self.current_epoch = 0
        print("sampler_steps={} lengths={}".format(self.sampler_steps, self.lengths))

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        min_len = 1e5
        max_len = -1
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            if vid_len == 4 :
                print(vid)
            if min_len > vid_len :
                min_len = vid_len
            if max_len < vid_len :
                max_len = vid_len

            for exp_id, exp_dict in vid_data['expressions'].items():
                
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = int(exp_dict['obj_id'])
                meta['frames'] = vid_frames
                
                
                # get object category
                obj_id = exp_dict['obj_id']
                meta['category'] = vid_meta['objects'][obj_id]['category']
                self.metas.append(meta)
        print("min : ", min_len)
        print("max : ", max_len)
        #self.metas = self.metas[0:100]


    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        
        meta = self.metas[idx]  # dict

        video, exp, obj_id, category, frames = meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames']

        # clean up the caption
        #print(exp)
        exp = " ".join(exp.lower().split())
        #print(exp, len(exp), type(exp))

        category_id = category_dict[category]
        vid_len = len(frames)


        num_frames = self.num_frames

        if vid_len < num_frames :
            if num_frames>2*vid_len :
                k = num_frames//vid_len
                sample_indx = []
                for j in range(k) :
                    sample_indx = sample_indx + list(range(vid_len))

                sample_indx = sample_indx + list(range(num_frames%vid_len))
            else :
                sample_indx = list(range(vid_len)) + list(range(num_frames-vid_len))
        else :
            sample_indx = random.sample(list(range(vid_len)), num_frames)

        #print("------>", vid_len)
        
        sample_indx.sort()

        print([frames[i] for i in sample_indx])
    

        # read frames and masks
        imgs, masks, valid, captions = [], [], [], []
        for frame_indx in sample_indx:
            
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
            mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('P')
            #print(np.array(img).shape)


            # create the target
            mask = np.array(mask)
            mask = (mask == obj_id).astype(np.uint8)  # 0,1 binary
            mask = Image.fromarray(mask, mode="P")

            img, mask = self._transforms(img, mask)

           

            if (mask > 0).any():
                valid.append(1)
            else:  # some frame didn't contain the instance
                valid.append(0)

            # append
            imgs.append(img)
            masks.append(mask)
            #captions.append(exp)
            
        
        masks = torch.stack(masks, dim=0)
        imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
        target = {
            #'frames_idx': torch.tensor(sample_indx),  # [T,]
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'caption': exp,
            'dname': 'ytvos'
        }
        

        return imgs, target















def build(image_set, args):
    print("building image level YTVOS dataset")
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
    }
    img_folder, ann_file = PATHS[image_set]

    if args.half :

        dataset = YTVOSDatasetVid(img_folder, ann_file, transforms=im_transform256,
                            return_masks=args.masks,
                            num_frames=args.num_frames, max_skip=args.max_skip, sampler_interval=args.sampler_interval,
                            args=args)
        
    elif args.cogdres :

        dataset = YTVOSDatasetVid(img_folder, ann_file, transforms=im_transformcog,
                            return_masks=args.masks,
                            num_frames=args.num_frames, max_skip=args.max_skip, sampler_interval=args.sampler_interval,
                            args=args)

    elif args.wanres :

        dataset = YTVOSDatasetVid(img_folder, ann_file, transforms=im_transformwan,
                            return_masks=args.masks,
                            num_frames=args.num_frames, max_skip=args.max_skip, sampler_interval=args.sampler_interval,
                            args=args)

    
    else :

        dataset = YTVOSDatasetVid(img_folder, ann_file, transforms=im_transform,
                            return_masks=args.masks,
                            num_frames=args.num_frames, max_skip=args.max_skip, sampler_interval=args.sampler_interval,
                            args=args)

    
    
    return dataset





if __name__ == "__main__" :


    import sys
    sys.path.insert(1, "../")
    import opts

    print("deps---resolved")
    parser = argparse.ArgumentParser('OnlineRefer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    print(args.__dict__)
    dataset = build("train", args)

