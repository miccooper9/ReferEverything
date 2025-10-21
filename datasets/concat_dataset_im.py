from pathlib import Path


import torch.utils.data

from torch.utils.data import ConcatDataset
from .refexp2imgseq import build as build_seq_refexp


def build(image_set, args):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    for name in coco_names:
        coco_seq =  build_seq_refexp(name, image_set, args)
        print(f"-----{name} : ", len(coco_seq))

        concat_data.append(coco_seq)

    
    #print(llll)

    concat_data = ConcatDataset(concat_data)

    return concat_data
