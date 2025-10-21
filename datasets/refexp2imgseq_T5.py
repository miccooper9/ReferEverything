
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import random
import numpy as np
from PIL import Image

import os


import datasets.transforms as Tr



class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, T5_enc_path, num_frames, transforms, return_masks):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.T5_enc_path = T5_enc_path
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_frames = num_frames
        '''self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-20, 20), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))'''
        print("-----img2seq-----")

    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks

    def __getitem__(self, idx):
        
        
        img, target = super(ModulatedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]


        uid = coco_img["id"]
        origid = coco_img["original_id"]
        vid = coco_img["file_name"].split("_")[-1].split(".")[0]


        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None

        pemb_pth = os.path.join(self.T5_enc_path, 'rcoco', dataset_name, f'{vid}_{uid}_{origid}_nonull_temb.pt')
        pemb = torch.load(pemb_pth, map_location="cpu")


        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)


        im = Image.fromarray(np.uint8(img))

        numpy_masks = target['masks'].numpy() # [1, H, W]
        
        assert numpy_masks.shape[0] == 1

        msk = Image.fromarray(numpy_masks[0], mode="P")

        im, msk = self._transforms(im, msk)



        seq_images, seq_instance_masks = [im,], [msk]
        valid=[]

        if (msk > 0).any():
            valid.append(1)
        else:  # some frame didn't contain the instance
            valid.append(0)


        numinst = len(numpy_masks)
        assert numinst == 1
        for t in range(self.num_frames - 1):

            

            if (msk > 0).any():
                valid.append(1)
            else:  # some frame didn't contain the instance
                valid.append(0)

            seq_images.append(im)
            seq_instance_masks.append(msk)

       

           
        output_inst_masks = torch.stack(seq_instance_masks, dim=0)            # [t, h, w]
        frames = torch.stack(seq_images,dim=0)
        
       

        out_target = {
            #'frames_idx': torch.tensor(sample_indx),  # [T,]
            'masks': output_inst_masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'caption': caption,
            'p_emb': pemb[0],
            'dname': dataset_name,
        }

        
        return frames, out_target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # xminyminwh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        #classes = [obj["category_id"] for obj in anno]
        #classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # keep the valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        #classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        #target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target





im_transform = Tr.Compose([
    Tr.Resize(512,512),
    Tr.ToTensor(),
    Tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def build(dataset_file, image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    dataset = dataset_file
    PATHS = {
        "train": (root / "train2014", root / dataset / f"{mode}_{dataset}_train.json"),
        "val": (root / "train2014", root / dataset / f"{mode}_{dataset}_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = ModulatedDetection(
        img_folder,
        ann_file,
        args.T5_enc_path,
        num_frames=args.num_frames,
        transforms=im_transform,
        return_masks=True,#args.masks,
    )
    return dataset




if __name__ == "__main__" :


    import sys
    sys.path.insert(1, "../")
    import opts
    import argparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os


    print("deps---resolved")
    parser = argparse.ArgumentParser('OnlineRefer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    print(args.__dict__)
    dataset = build("refcoco+", "train", args)


    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)

    for idx,batch in enumerate(loader):

        img, target = batch
        print(target['caption'])
        print(img.shape, target['masks'].shape)
        print(target['valid'])
        print(target['dname'])


            

