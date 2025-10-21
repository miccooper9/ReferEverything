
import sys
sys.path.insert(1, "../")

import argparse

import json
import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
import os

import torchvision.transforms as T

import opts

from collections import defaultdict
from pycocotools.mask import decode

# build transform
transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



transform256 = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])




def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j





def evaluate(result_dir='') :

    with open(f'{args.TAO_path}/annotations_burst/val/all_classes.json') as f :

        val_dict  = json.load(f)

    with open(f'{args.TAO_path}/annotations_burst/test/all_classes.json') as f :

        test_dict  = json.load(f)

    root = Path(f'{args.TAO_path}/frames/val')  # data/ref-davis
    
    seqs  = test_dict['sequences'] + val_dict['sequences']

    cats = val_dict['categories']
    id2cls = {}
    for c in cats :
        id2cls[c['id']] = c['name']

    vid2cat = defaultdict(dict)
    for seq in seqs :

        local = defaultdict(list)

        for t,c in seq['track_category_ids'].items() :
            local[c].append(t)

        vid2cat[seq['seq_name']] = local

    vid2cagg = defaultdict(dict)

    for seq in tqdm(seqs):

        vname = seq['seq_name']

        c_id2trk  = vid2cat[vname]

        cid2agg = defaultdict(list)

        segs = seq['segmentations']



        for c_id, tlist in c_id2trk.items() :


            c_agg = [[] for i in range(len(segs))]

            for t in tlist :

                for f in range(len(segs)) :

                    fseg = segs[f]

                    if str(t) in fseg :
                        c_agg[f].append(fseg[str(t)]['rle'])


            cid2agg[c_id] = c_agg

        
        vid2cagg[vname] = cid2agg


    


    cls2iou = defaultdict(list)

    for seq in tqdm(seqs):

        video_name = seq['seq_name']
        data_src = seq['dataset']
        frames = seq['annotated_image_paths']
        csegs  = vid2cagg[video_name]
        img_size = seq["height"], seq["width"]

        print("=====", video_name, "====", csegs.keys())
        print(frames)

        #if seq['dataset'] in ['HACS', 'AVA'] :
        #    continue


        for c,cl in csegs.items() :

            cls_name = id2cls[c]

            c_preds = []
            c_gts = []
            imgs = []
            
            for i in range(len(frames)) :

                print("frame :",i )

                frame = frames[i]
                fsegs = cl[i]

                f_gt  = np.zeros(img_size)

                mask_path = os.path.join(result_dir, data_src, video_name, 'pred', f'{c}_{cls_name}',frame.split('.')[0] + '.png')
                f_pred  = np.array(Image.open(mask_path))

                c_preds.append(f_pred)





                for seg in fsegs :

                    gt_mask = decode({
                        "size": img_size,
                        "counts": seg.encode("utf-8")
                    })
                    

                    f_gt[gt_mask==1] = 1

                c_gts.append(f_gt)


            c_gts = np.stack(c_gts,axis=0)
            c_preds = np.stack(c_preds,axis=0)
            
            

            print("c_id : ",c_gts.shape, c_preds.shape)

            iou = db_eval_iou(c_gts,c_preds)
            print(cls_name, ">>>>>>",np.mean(iou), iou)

            cls2iou[cls_name].append(np.mean(iou))

    

    overall_j =  []

    for cls,j_list in cls2iou.items() :

        
        j = np.mean(np.array(j_list))
        overall_j.append(j)
        print(cls, " ", j)


    print("overall_j : ", np.mean(np.array(overall_j)))

                

            


    return




def main(args):

    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    JFm = evaluate(result_dir=args.result_dir)


    
    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json

    main(args)



