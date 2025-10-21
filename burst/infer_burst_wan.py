import sys
sys.path.insert(1, "../")

import argparse
import datetime
import json
import random

from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from PIL import Image




from tqdm import tqdm
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.want2v_wrapper import  build_want2vlarge_txt_model, WanVdiff_wrapper

import opts
from collections import defaultdict

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

def evaluate(args, model) :


    with open('./input_frame_paths.json') as f :
        inf_frames = json.load(f)


    with open(f'{args.TAO_path}/annotations_burst/val/all_classes.json') as f :

        val_dict  = json.load(f)

    with open(f'{args.TAO_path}/annotations_burst/test/all_classes.json') as f :

        test_dict  = json.load(f)

    root_val = Path(f'{args.TAO_path}/frames/val')  # data/ref-davis
    root_test = Path(f'{args.TAO_path}/frames/test')

    for s in val_dict['sequences'] :
        s["root_path"] = root_val 

    for s in test_dict['sequences'] :
        s["root_path"] = root_test
    
    seqs  = test_dict['sequences'] + val_dict['sequences']


    cats = val_dict['categories']
    id2cls = {}
    for c in cats :
        id2cls[c['id']] = c['name']

    vid2cat = defaultdict(list)
    for seq in seqs :

        local = set()

        for t,c in seq['track_category_ids'].items() :
            local.add(c)

        vid2cat[seq['seq_name']] = list(local)
    

    save_path_prefix = os.path.join(args.val_vis, f'burst_valtest')

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # get palette
    palette_img_path = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette_img = Image.open(palette_img_path)
    palette = palette_img.getpalette()

    
    for seq in tqdm(seqs):

        video_name = seq['seq_name']
        data_src = seq['dataset']
        root = seq["root_path"]

        for c_id in vid2cat[video_name] :

            with torch.no_grad() :


                cls_name = id2cls[c_id]

        
                exp = f'the {cls_name}'

                frames = inf_frames[f"{data_src}_{video_name}"]#seq['all_image_paths']#seq['annotated_image_paths']
                fanno = seq["annotated_image_paths"]
                video_len = len(frames)
                print(f"{exp} video_len {video_len}")

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



                fnums = [65,49,37,25,17,9]
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
                        img_path = os.path.join(root, data_src, video_name, frame)
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        print(img_path,"--------", np.array(img).shape, origin_w, origin_h)
                        #print(palette_img_path, np.array(palette_img).shape)
                        if args.half :
                            tim = transform256(img).to(args.device)  # list[Img]
                        else :
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
                    vid_masks.append(pred.cpu())
                    f_ims.append(rimgs.cpu())


                vid_masks = torch.cat(vid_masks, dim=0).cpu()
                f_ims = torch.cat(f_ims, dim=0).cpu()
                print(video_len,"-------->>>>", vid_masks.shape, f_ims.shape)

                anno_save_path_pred = os.path.join(save_path_prefix, data_src, video_name, "pred", f'{c_id}_{cls_name}')#exp.replace(" ", "_"))
                print("anno_save_path_pred : ",anno_save_path_pred)

                if not os.path.exists(anno_save_path_pred):
                    os.makedirs(anno_save_path_pred)


                
                annoset = set([f.split('.')[0] for f in fanno])
                anno_cnt = 0
                    
                for f in range(vid_masks.shape[0]):
                    #fig, ax = plt.subplots(2, 1, figsize=(20, 30))


                    

                    f_id = frames[f].split('.')[0]

                    if f_id not in annoset :
                        continue
                    
                    anno_cnt+=1

                    
                    img_E = Image.fromarray(vid_masks[f].detach().cpu().numpy().astype(np.uint8))
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(anno_save_path_pred, f'{f_id}.png'))
                    


                assert anno_cnt == len(annoset)

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



