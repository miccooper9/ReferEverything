import sys
sys.path.insert(1, "../")

import argparse
import datetime
import json
import random

from datetime import datetime
import numpy as np
import torch

from PIL import Image



from tqdm import tqdm
import os

import torch.nn.functional as F
import torchvision.transforms as T


from models.want2v_wrapper import  build_want2vlarge_txt_model, WanVdiff_wrapper

import opts
#from tensorboardX import SummaryWriter

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






def evaluate(args, model) :



    vDIR= args.vspw_path#sys.argv[1]
    split = 'val.txt'

    with open(os.path.join(vDIR,split),'r') as f:
        lines = f.readlines()
        video_list = [line[:-1] for line in lines]


    save_path_prefix = os.path.join(args.val_vis, f'vspw_val')

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # get palette
    palette_img_path = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette_img = Image.open(palette_img_path)
    palette = palette_img.getpalette()


    with open('./val_vid2classes.json') as f :
        val_vid2classes = json.load(f)
    

    max_token_len = -1

    with torch.no_grad() :
    
        for video in tqdm(video_list):

            
            frames = sorted(os.listdir(f'{vDIR}/data/{video}/origin'))
            

            video_len = len(frames)
            print(f"{video} video_len {video_len}")
            # NOTE: the im2col_step for MSDeformAttention is set as 64
            # so the max length for a clip is 64
            # store the video pred results
            all_pred_logits = []
            all_pred_masks = []


            cur_stuff_classes = val_vid2classes[video]

            for i in range(len(cur_stuff_classes)) :

                

                class_idx = int(cur_stuff_classes[i])

                if class_idx not in stuffid2name :
                    continue

                

                class_name = stuffid2name[class_idx]


                exp = f"the {class_name}"

                text_in = model.backbone.tokenizer(
                    exp,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )


                text_input_ids, text_mask = text_in.input_ids, text_in.attention_mask


                prompt_embeds = model.backbone.text_encoder(text_input_ids.to(args.gpu), text_mask.to(args.gpu)).last_hidden_state

                seq_lens = text_mask.gt(0).sum(dim=1).long()
                print("seq_lens : ", seq_lens)
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

            
                prompt_embeds = torch.stack(
                    [torch.cat([u, u.new_zeros(args.token_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
                )

                print("prompt embeds : ", prompt_embeds.shape, prompt_embeds.dtype)

                prompt_embeds = prompt_embeds.to(args.device, model.backbone.dit.dtype)

                print("prompt embeds : ", prompt_embeds.shape, prompt_embeds.dtype)
                
                num_clip_frames = args.num_frames


                fnums = [105,73,61,49,37,25,17,9]
                fnumidx = fnums.index(num_clip_frames)

                while video_len < num_clip_frames :
                    
                    fnumidx+=1
                    num_clip_frames = fnums[fnumidx]


                print("video len : ", video_len, " num_clip_frames : ", num_clip_frames, " argsF : ", args.num_frames)
                
                
                # 3. for each clip
                vid_masks = []
                f_ims = []
                last_flag = False
                for clip_id in range(0, video_len, num_clip_frames):
                    frames_ids = [x for x in range(video_len)]
                    clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                    #print(clip_frames_ids)
                    clip_len = len(clip_frames_ids)

                    print("======clip len : ", clip_len)
                                    

                    if clip_len < num_clip_frames :
                        last_flag =  True
                        clip_frames_ids = frames_ids[-num_clip_frames : ]
                        
                        print("2======clip len : ", clip_len)


                    
                    # load the clip images
                    pframes = []
                    for t in clip_frames_ids:
                        frame = frames[t]
                        img_path = os.path.join(vDIR, "data", video, "origin",frame)
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        print(img_path,"--------", np.array(img).shape, origin_w, origin_h)
                        #print(palette_img_path, np.array(palette_img).shape)
                        
                        tim = transform(img).to(args.device)  # list[Img]
                        pframes.append(tim)
                        
                    imgs = torch.stack(pframes, dim=0).to(args.device)
                    print(f"{exp} img shape {imgs.shape} {video_len}")

                    imgs = imgs.to(args.device,model.backbone.vae.dtype)


                    img_latents = model.backbone.vae.encode(imgs.unsqueeze(0).permute(0,2,1,3,4)).latent_dist.mode()
                    latents_mean = (torch.tensor(model.backbone.vae.config.latents_mean).view(1, model.backbone.vae.config.z_dim, 1, 1, 1).to(args.device, model.backbone.vae.dtype))
                    latents_std = 1.0 / torch.tensor(model.backbone.vae.config.latents_std).view(1, model.backbone.vae.config.z_dim, 1, 1, 1).to(args.device, model.backbone.vae.dtype)
                    img_latents = (img_latents - latents_mean) * latents_std

                    print("img_latents : ", img_latents.shape, img_latents.max(), img_latents.min())
                    print("p embeds : ", prompt_embeds.shape)

                    pred_latents = model(img_latents.to(model.backbone.dit.dtype), timesteps=0, prompt_embeds=prompt_embeds)
        
                    pred_latents = pred_latents.to(model.backbone.vae.dtype) / latents_std + latents_mean
                    seg_masks = model.backbone.vae.decode(pred_latents, return_dict=False)[0] #1, 3, t ,h, w


                    if last_flag :
                        seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)[-clip_len : , ...]
                        imgs = imgs[-clip_len : , ...]
                    else :
                        seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)
                    
                    print("seg_masks : ", seg_masks.shape, seg_masks.max(), seg_masks.min()) 

                    
                    pred_masks = F.interpolate(seg_masks, size=(origin_h, origin_w), mode='bilinear',
                                            align_corners=False)# t, c, h , w
                    rimgs = F.interpolate(imgs, size=(origin_h, origin_w), mode='bilinear',
                                                align_corners=False)
                
                    pred = pred_masks.mean(1)
                    pred[pred>0.5] = 1.0
                    pred[pred<=0.5] = 0.0
                    print("---------pred :", pred.shape, pred.max(), pred.min())
                    vid_masks.append(pred)
                    f_ims.append(rimgs)


                vid_masks = torch.cat(vid_masks, dim=0).cpu()
                f_ims = torch.cat(f_ims, dim=0).cpu()
                print(video_len,"-------->>>>", vid_masks.shape, f_ims.shape)

                anno_save_path_pred = os.path.join(save_path_prefix, video, class_name, "pred")#exp.replace(" ", "_"))
                
                print("anno_save_path_pred : ",anno_save_path_pred)
                
                if not os.path.exists(anno_save_path_pred):
                    os.makedirs(anno_save_path_pred)


                for f in tqdm(range(vid_masks.shape[0])):
                
                    f_id = frames[f].split('.')[0]

                    
                    img_E = Image.fromarray(vid_masks[f].detach().cpu().numpy().astype(np.uint8))
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(anno_save_path_pred, f'{f_id}.png'))
                    


    return



def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path = (f"{args.output_dir}/{name_prefix}")
    log_path = os.path.join(exp_path, 'log')

    val_vis = os.path.join(exp_path, 'val_vis')
    if not os.path.exists(log_path): 
        os.makedirs(log_path)

    if not os.path.exists(val_vis): 
        os.makedirs(val_vis)
    
    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, val_vis





def main(args):

    
    args.gpu = args.device
    print("device : ", args.gpu)

    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    backbone = build_want2vlarge_txt_model(args.base_dir)#VDiffFeatExtractor()    
    model = WanVdiff_wrapper(backbone)
    model.to(args.gpu)

    




    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=False)
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded ", resume_epoch)
        
    else:
        print("no checkpoint---------")
        resume_epoch = -999



    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
        
        evaluate(args, model)


        

    
    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    args.log_path, args.val_vis = set_path(args)

    main(args)



