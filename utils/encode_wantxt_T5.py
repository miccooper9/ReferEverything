import sys
sys.path.insert(1, "../")

import argparse

import json
import random

from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
import os


import torchvision.transforms as T



import opts



from models.want2v_wrapper import  build_want2v_txt
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

import torchvision




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


class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, num_frames, transforms, return_masks):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        

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

        

        return coco_img










def encode_rcoco(args, text_encoder, tokenizer) :
        


    for dfile in ["refcoco", "refcoco+", "refcocog"] :

        save_path = f"/mnt/fsx/Anurag/Wan_pre_encoded/txtT5/rcoco/{dfile}"


        root = Path(args.coco_path)
        assert root.exists(), f"provided COCO path {root} does not exist"
        mode = "instances"
        dataset = dfile
        PATHS = {
            "train": (root / "train2014", root / dataset / f"{mode}_{dataset}_train.json"),
            "val": (root / "train2014", root / dataset / f"{mode}_{dataset}_val.json"),
        }

        img_folder, ann_file = PATHS["train"]


        dataset = ModulatedDetection(img_folder,ann_file,num_frames=args.num_frames,transforms=transform,return_masks=True)


        print(len(dataset))


        for i in tqdm(range(len(dataset))) :
            
            coco_img = dataset[i]
            uid = coco_img["id"]
            origid = coco_img["original_id"]
            prompts = [coco_img["caption"]]
            vid = coco_img["file_name"].split("_")[-1].split(".")[0]


            prompts = [prompt_clean(u) for u in prompts]


            try :

                text_in = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                text_input_ids, text_mask = text_in.input_ids, text_in.attention_mask

                
                with torch.no_grad() :

                    prompt_embeds = text_encoder(text_input_ids.to(args.gpu), text_mask.to(args.gpu)).last_hidden_state

                seq_lens = text_mask.gt(0).sum(dim=1).long()
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

            
                prompt_embeds = torch.stack( [u for u in prompt_embeds], dim=0)


                prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=args.device)
                
            
                torch.save(prompt_embeds, f"{save_path}/{vid}_{uid}_{origid}_nonull_temb.pt")


            except :
                print(f"encoding {vid} {uid} {origid} failed")

            


        






def encode_rdavis(args, text_encoder, tokenizer) :



    root = Path(args.davis_path)  # data/ref-davis
    
    meta_file = os.path.join(root, "meta_expressions", 'valid', "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    video_list = list(data.keys())

    save_path = '/mnt/fsx/Anurag/Wan_pre_encoded/txtT5/rdavis'

    for video in tqdm(video_list):

        

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])


        for i in range(num_expressions):

            
            vid = video
            exp = expressions[expression_list[i]]["exp"]
            exp_id = expression_list[i]  # start from 0
            o_id = expressions[expression_list[i]]["obj_id"]
            


            prompts = [exp]
            prompts = [prompt_clean(u) for u in prompts]
            try :

                text_in = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                text_input_ids, text_mask = text_in.input_ids, text_in.attention_mask

                
                with torch.no_grad() :

                    prompt_embeds = text_encoder(text_input_ids.to(args.gpu), text_mask.to(args.gpu)).last_hidden_state

                seq_lens = text_mask.gt(0).sum(dim=1).long()
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

            
                prompt_embeds = torch.stack( [u for u in prompt_embeds], dim=0)


                prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=args.device)

                
                
                torch.save(prompt_embeds, f"{save_path}/{vid}_{exp_id}_{o_id}_nonull_temb.pt")
                

            except :
                print(f"encoding {vid}_{exp_id}_{o_id} failed")

    













def encode_rvos(args, text_encoder, tokenizer):

    
    train_anno = '/mnt/fsx/Anurag/rvos/meta_expressions/train/meta_expressions.json'
    save_path = f'{args.output_dir}/txtT5/rvos'

    with open(str(train_anno), 'r') as f:
        anno = json.load(f)['videos']


    videos = list(anno.keys())

    for vid in tqdm(videos):
        
        
        vid_data = anno[vid]
        
        for exp_id, exp_dict in tqdm(vid_data['expressions'].items()):
            
            meta = {}
            meta['video'] = vid
            exp = exp_dict['exp']
            o_id = int(exp_dict['obj_id'])

        
            prompts = [exp]
            prompts = [prompt_clean(u) for u in prompts]
            try :

                text_in = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                text_input_ids, text_mask = text_in.input_ids, text_in.attention_mask

                
                with torch.no_grad() :

                    prompt_embeds = text_encoder(text_input_ids.to(args.gpu), text_mask.to(args.gpu)).last_hidden_state

                seq_lens = text_mask.gt(0).sum(dim=1).long()
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

            
                prompt_embeds = torch.stack( [u for u in prompt_embeds], dim=0)


                prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=args.device)


                
            
                torch.save(prompt_embeds, f"{save_path}/{vid}_{exp_id}_{o_id}_nonull_temb.pt")
                

            except :
                print(f"encoding {vid}_{exp_id}_{o_id} failed")


                   


            
                        

                            


        


            

            
    


def main(args):

    

    

    print("device : ", args.device)
    args.gpu = args.device


    

    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    






    tokenizer, txt_enc = build_want2v_txt(args.base_dir)

    
    txt_enc.to(args.device)
    #tokenizer.to(args.device)
    

    print("==============================================================================ENCODING RVOS==============================================================================")
    encode_rvos(args, txt_enc, tokenizer)
    print("==============================================================================ENCODING RDAVIS==============================================================================")
    encode_rdavis(args, txt_enc, tokenizer)
    print("==============================================================================ENCODING REFCOCO==============================================================================")
    encode_rcoco(args, txt_enc, tokenizer)

    


        

    

    return
    

    



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))

    main(args)



