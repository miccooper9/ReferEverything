
import sys
sys.path.insert(1, "../")

import argparse

import random


import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
import os

import torchvision.transforms as T

import opts

from collections import defaultdict
from davis2017.metrics import db_eval_iou


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



stuff_classes = [{'id': 0, 'name': 'wall', 'isthing': 0, 'color': [120, 120, 120]}, {'id': 1, 'name': 'ceiling', 'isthing': 0, 'color': [180, 120, 120]}, {'id': 3, 'name': 'stair', 'isthing': 0, 'color': [80, 50, 50]}, {'id': 5, 'name': 'escalator', 'isthing': 0, 'color': [120, 120, 80]}, {'id': 6, 'name': 'Playground_slide', 'isthing': 0, 'color': [140, 140, 140]}, {'id': 7, 'name': 'handrail_or_fence', 'isthing': 0, 'color': [204, 5, 255]}, {'id': 9, 'name': 'rail', 'isthing': 0, 'color': [4, 250, 7]}, {'id': 11, 'name': 'pillar', 'isthing': 0, 'color': [235, 255, 7]}, {'id': 12, 'name': 'pole', 'isthing': 0, 'color': [150, 5, 61]}, {'id': 13, 'name': 'floor', 'isthing': 0, 'color': [120, 120, 70]}, {'id': 14, 'name': 'ground', 'isthing': 0, 'color': [8, 255, 51]}, {'id': 15, 'name': 'grass', 'isthing': 0, 'color': [255, 6, 82]}, {'id': 16, 'name': 'sand', 'isthing': 0, 'color': [143, 255, 140]}, {'id': 17, 'name': 'athletic_field', 'isthing': 0, 'color': [204, 255, 4]}, {'id': 18, 'name': 'road', 'isthing': 0, 'color': [255, 51, 7]}, {'id': 19, 'name': 'path', 'isthing': 0, 'color': [204, 70, 3]}, {'id': 20, 'name': 'crosswalk', 'isthing': 0, 'color': [0, 102, 200]}, {'id': 21, 'name': 'building', 'isthing': 0, 'color': [61, 230, 250]}, {'id': 22, 'name': 'house', 'isthing': 0, 'color': [255, 6, 51]}, {'id': 23, 'name': 'bridge', 'isthing': 0, 'color': [11, 102, 255]}, {'id': 24, 'name': 'tower', 'isthing': 0, 'color': [255, 7, 71]}, {'id': 25, 'name': 'windmill', 'isthing': 0, 'color': [255, 9, 224]}, {'id': 26, 'name': 'well_or_well_lid', 'isthing': 0, 'color': [9, 7, 230]}, {'id': 27, 'name': 'other_construction', 'isthing': 0, 'color': [220, 220, 220]}, {'id': 28, 'name': 'sky', 'isthing': 0, 'color': [255, 9, 92]}, {'id': 29, 'name': 'mountain', 'isthing': 0, 'color': [112, 9, 255]}, {'id': 30, 'name': 'stone', 'isthing': 0, 'color': [8, 255, 214]}, {'id': 31, 'name': 'wood', 'isthing': 0, 'color': [7, 255, 224]}, {'id': 32, 'name': 'ice', 'isthing': 0, 'color': [255, 184, 6]}, {'id': 33, 'name': 'snowfield', 'isthing': 0, 'color': [10, 255, 71]}, {'id': 34, 'name': 'grandstand', 'isthing': 0, 'color': [255, 41, 10]}, {'id': 35, 'name': 'sea', 'isthing': 0, 'color': [7, 255, 255]}, {'id': 36, 'name': 'river', 'isthing': 0, 'color': [224, 255, 8]}, {'id': 37, 'name': 'lake', 'isthing': 0, 'color': [102, 8, 255]}, {'id': 38, 'name': 'waterfall', 'isthing': 0, 'color': [255, 61, 6]}, {'id': 39, 'name': 'water', 'isthing': 0, 'color': [255, 194, 7]}, {'id': 40, 'name': 'billboard_or_Bulletin_Board', 'isthing': 0, 'color': [255, 122, 8]}, {'id': 42, 'name': 'pipeline', 'isthing': 0, 'color': [255, 8, 41]}, {'id': 45, 'name': 'cushion_or_carpet', 'isthing': 0, 'color': [235, 12, 255]}, {'id': 53, 'name': 'wheeled_machine', 'isthing': 0, 'color': [255, 224, 0]}, {'id': 57, 'name': 'tyre', 'isthing': 0, 'color': [0, 235, 255]}, {'id': 58, 'name': 'traffic_light', 'isthing': 0, 'color': [0, 173, 255]}, {'id': 59, 'name': 'lamp', 'isthing': 0, 'color': [31, 0, 255]}, {'id': 66, 'name': 'tree', 'isthing': 0, 'color': [255, 0, 0]}, {'id': 67, 'name': 'flower', 'isthing': 0, 'color': [255, 163, 0]}, {'id': 68, 'name': 'other_plant', 'isthing': 0, 'color': [255, 102, 0]}, {'id': 69, 'name': 'toy', 'isthing': 0, 'color': [194, 255, 0]}, {'id': 70, 'name': 'ball_net', 'isthing': 0, 'color': [0, 143, 255]}, {'id': 71, 'name': 'backboard', 'isthing': 0, 'color': [51, 255, 0]}, {'id': 73, 'name': 'bat', 'isthing': 0, 'color': [0, 255, 41]}, {'id': 75, 'name': 'cupboard_or_showcase_or_storage_rack', 'isthing': 0, 'color': [10, 0, 255]}, {'id': 80, 'name': 'trash_can', 'isthing': 0, 'color': [255, 0, 245]}, {'id': 81, 'name': 'cage', 'isthing': 0, 'color': [255, 0, 102]}, {'id': 93, 'name': 'shelf', 'isthing': 0, 'color': [51, 0, 255]}, {'id': 94, 'name': 'bathtub', 'isthing': 0, 'color': [0, 194, 255]}, {'id': 98, 'name': 'other_machine', 'isthing': 0, 'color': [0, 255, 10]}, {'id': 103, 'name': 'curtain', 'isthing': 0, 'color': [255, 235, 0]}, {'id': 104, 'name': 'textiles', 'isthing': 0, 'color': [8, 184, 170]}, {'id': 105, 'name': 'clothes', 'isthing': 0, 'color': [133, 0, 255]}, {'id': 110, 'name': 'book', 'isthing': 0, 'color': [0, 214, 255]}, {'id': 111, 'name': 'tool', 'isthing': 0, 'color': [255, 0, 112]}, {'id': 112, 'name': 'blackboard', 'isthing': 0, 'color': [92, 255, 0]}, {'id': 113, 'name': 'tissue', 'isthing': 0, 'color': [0, 224, 255]}, {'id': 119, 'name': 'other_electronic_product', 'isthing': 0, 'color': [255, 0, 163]}, {'id': 120, 'name': 'fruit', 'isthing': 0, 'color': [255, 204, 0]}, {'id': 121, 'name': 'food', 'isthing': 0, 'color': [255, 0, 143]}]
stuffid2name =  {0: 'wall', 1: 'ceiling', 3: 'stair', 5: 'escalator', 6: 'Playground_slide', 7: 'handrail_or_fence', 9: 'rail', 11: 'pillar', 12: 'pole', 13: 'floor', 14: 'ground', 15: 'grass', 16: 'sand', 17: 'athletic_field', 18: 'road', 19: 'path', 20: 'crosswalk', 21: 'building', 22: 'house', 23: 'bridge', 24: 'tower', 25: 'windmill', 26: 'well_or_well_lid', 27: 'other_construction', 28: 'sky', 29: 'mountain', 30: 'stone', 31: 'wood', 32: 'ice', 33: 'snowfield', 34: 'grandstand', 35: 'sea', 36: 'river', 37: 'lake', 38: 'waterfall', 39: 'water', 40: 'billboard_or_Bulletin_Board', 42: 'pipeline', 45: 'cushion_or_carpet', 53: 'wheeled_machine', 57: 'tyre', 58: 'traffic_light', 59: 'lamp', 66: 'tree', 67: 'flower', 68: 'other_plant', 69: 'toy', 70: 'ball_net', 71: 'backboard', 73: 'bat', 75: 'cupboard_or_showcase_or_storage_rack', 80: 'trash_can', 81: 'cage', 93: 'shelf', 94: 'bathtub', 98: 'other_machine', 103: 'curtain', 104: 'textiles', 105: 'clothes', 110: 'book', 111: 'tool', 112: 'blackboard', 113: 'tissue', 119: 'other_electronic_product', 120: 'fruit', 121: 'food'}
stuffname2id = {'wall': 0, 'ceiling': 1, 'stair': 3, 'escalator': 5, 'Playground_slide': 6, 'handrail_or_fence': 7, 'rail': 9, 'pillar': 11, 'pole': 12, 'floor': 13, 'ground': 14, 'grass': 15, 'sand': 16, 'athletic_field': 17, 'road': 18, 'path': 19, 'crosswalk': 20, 'building': 21, 'house': 22, 'bridge': 23, 'tower': 24, 'windmill': 25, 'well_or_well_lid': 26, 'other_construction': 27, 'sky': 28, 'mountain': 29, 'stone': 30, 'wood': 31, 'ice': 32, 'snowfield': 33, 'grandstand': 34, 'sea': 35, 'river': 36, 'lake': 37, 'waterfall': 38, 'water': 39, 'billboard_or_Bulletin_Board': 40, 'pipeline': 42, 'cushion_or_carpet': 45, 'wheeled_machine': 53, 'tyre': 57, 'traffic_light': 58, 'lamp': 59, 'tree': 66, 'flower': 67, 'other_plant': 68, 'toy': 69, 'ball_net': 70, 'backboard': 71, 'bat': 73, 'cupboard_or_showcase_or_storage_rack': 75, 'trash_can': 80, 'cage': 81, 'shelf': 93, 'bathtub': 94, 'other_machine': 98, 'curtain': 103, 'textiles': 104, 'clothes': 105, 'book': 110, 'tool': 111, 'blackboard': 112, 'tissue': 113, 'other_electronic_product': 119, 'fruit': 120, 'food': 121}

def evaluate(result_dir='') :


    video_list = sorted(os.listdir(result_dir))

    


    print(len(video_list))
    print(video_list[0:5])

    stuff2vids = defaultdict(list)

    for video in video_list :

        v_stuffs = os.listdir(os.path.join(result_dir,video))
        #if video == '998_wFkstkiNp_I' :
        #    print("--------------->",v_stuffs)
        for stf in v_stuffs :
            stuff2vids[stf].append(video)

    
    print(stuff2vids)



    stuff_J = defaultdict(list)


    for stuff,vids in tqdm(stuff2vids.items()) :
        

        #print(">>>>>>>>>>>>>",stuff,vids)


        for vid in tqdm(vids) :

            ###### read GT

            mask_path = f'{args.vspw_path}/data/{vid}/mask/'

            mps = os.listdir(mask_path)

            gt_masks = []

            for mp in mps :

                msk = np.array(Image.open(f'{mask_path}/{mp}'))
                msk[msk == 0] = 255
                msk = msk - 1
                msk[msk == 254] = 124

                gt_msk = np.zeros(msk.shape)

                gt_msk[msk==stuffname2id[stuff]] = 1

                gt_masks.append(gt_msk)

            gt_masks = np.stack(gt_masks)[None,...]

            print(vid, "pred >>>>>" ,gt_masks.shape, gt_masks.max(), gt_masks.min())


            ####### read pred

            pred_path = f'{result_dir}/{vid}/{stuff}/pred'

            pfs = os.listdir(pred_path)

            res_masks = []

            for pf in pfs :

                pred = np.array(Image.open(f'{pred_path}/{pf}'))

                res_masks.append(pred)

            res_masks = np.stack(res_masks)[None,...]

            print(vid, "res >>>>>" ,res_masks.shape, res_masks.max(), res_masks.min())

            j_metrics_res = np.zeros((res_masks.shape[0], gt_masks.shape[0], gt_masks.shape[1]))
            for ii in range(gt_masks.shape[0]):
                for jj in range(res_masks.shape[0]):

                    j_metrics_res[jj, ii, :] = db_eval_iou(gt_masks[ii, ...], res_masks[jj, ...])

            J_m = np.mean(j_metrics_res)

            print(stuff, "J ::::",j_metrics_res.shape, J_m)
            stuff_J[stuff].append(J_m)


    overall_j =  []

    for stf,j_list in stuff_J.items() :

        
        j = np.mean(np.array(j_list))
        overall_j.append(j)
        print(stf, " ", j)


    print("overall_j : ", np.mean(np.array(overall_j)))



    return

    
    


            
    





def main(args):

    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    JFm = evaluate(result_dir=args.result_dir)#'/mnt/fsx-west/anurag/VSPW_stuff_evals/MS_jnt2x512_f32/val_vis/vspw_16')


    
    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json

    main(args)



