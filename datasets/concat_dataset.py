
from pathlib import Path

import torch
import torch.utils.data

from torch.utils.data import ConcatDataset
from .refexp2seq import build as build_seq_refexp

from .ytvos_clip import build as build_ytvos_clip

import random


def build(image_set, args):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    for name in coco_names:
        coco_seq =  build_seq_refexp(name, image_set, args)
        print(f"-----{name} : ", len(coco_seq))
        #subset from data
        inds = random.sample(range(1, len(coco_seq)), 4000)
        print(len(inds), inds[0:5])
        coco_sub = torch.utils.data.Subset(coco_seq, inds)
        concat_data.append(coco_sub)

    print('preparing ytvos dataset  .... ')
    ytvos_dataset = build_ytvos_clip(image_set, args)
    print(f"-----ytvos : ", len(ytvos_dataset))
    concat_data.append(ytvos_dataset)


    concat_data = ConcatDataset(concat_data)

    return concat_data