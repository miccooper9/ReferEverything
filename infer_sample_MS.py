
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
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
import wandb

from models.mst2v_wrapper import Vdiff_updown, build_vdiff_lean_updown
import opts

import cv2
from matplotlib.patches import Polygon

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


    frame_dir = os.path.join(args.frame_dir)
    exps = args.exps


    save_path_prefix = os.path.join(args.val_vis, f'chunked_by_f{args.num_frames}')

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # get palette
    palette_img_path = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette_img = Image.open(palette_img_path)
    palette = palette_img.getpalette()

    

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 5)]
    
    
    with torch.no_grad() :
        
            
        for exp in exps :

            frames = sorted(os.listdir(frame_dir))[10:30]
            video_len = len(frames)
            

            print(f"=======>{exp}")
            print(video_len, frames)


            text_in = model.backbone.tokenizer(
                exp,
                padding="max_length",
                max_length=model.backbone.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")


            prompt_embeds = model.backbone.text_encoder(
                            text_in.input_ids.to(args.device),
                            #attention_mask=text_in.attention_mask.to(args.device),
                            )[0]
            
            attention_mask = text_in['attention_mask'][:, :args.token_length]
            masks = ~attention_mask.to(torch.bool)

            print("prompt embeds : ", prompt_embeds.shape)
            #for k,v in text_in.items() :
            #    print(k,"----", v.shape)

            tokens = model.backbone.tokenizer.encode(exp)
            print(exp,len(tokens),tokens)
            print(attention_mask.shape, masks.shape)
            print(attention_mask[0,0:20])
            print(masks[0,0:20])




            prompt_embeds, masks, attention_mask = prompt_embeds[:,:args.token_length,:].to(args.device), masks.to(args.device), attention_mask.to(args.device)
            
            
            num_clip_frames = args.num_frames
            #prompt_in,_ = txtSA(prompt_embeds, mask=masks)
            #print("prompt_in : ", prompt_in.shape)
            
            
            
            # 3. for each clip
            vid_masks = []
            f_ims = []
            for clip_id in range(0, video_len, num_clip_frames):
                frames_ids = [x for x in range(video_len)]
                clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                #print(clip_frames_ids)
                clip_len = len(clip_frames_ids)


                
                # load the clip images
                pframes = []
                for t in clip_frames_ids:
                    frame = frames[t]
                    img_path = os.path.join(frame_dir, frame)
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


                pred_latents = model(imgs.unsqueeze(0),prompt_embeds)
                
                p_latents = 1 / model.backbone.vae.config.scaling_factor * pred_latents.detach()
                print("unscaled pred latents : ", p_latents.shape, p_latents.max(), p_latents.min())
                seg_masks = []
                for i in range(0,num_clip_frames,16) :
                    curr_latents = p_latents[i:i+16]
                    curr_seg = model.backbone.vae.decode(curr_latents).sample
                    seg_masks.append(curr_seg)
                    
                seg_masks = torch.cat(seg_masks, dim=0)
                
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



            anno_save_path_pred = os.path.join(save_path_prefix,  f'{exp.replace(" ", "_")}', "masks")#exp.replace(" ", "_"))
            anno_save_path_pred_orig = os.path.join(save_path_prefix, f'{exp.replace(" ", "_")}', "overlayed")
            print("anno_save_path_pred : ",anno_save_path_pred)
            print("anno_save_path_pred_orig : ",anno_save_path_pred_orig)

            if not os.path.exists(anno_save_path_pred):
                os.makedirs(anno_save_path_pred)
            if not os.path.exists(anno_save_path_pred_orig):
                os.makedirs(anno_save_path_pred_orig)



            for f in tqdm(range(vid_masks.shape[0])):
                #fig, ax = plt.subplots(2, 1, figsize=(20, 30))
                f_id = frames[f].split('.')[0] + '.png'

                
                img_E = Image.fromarray(vid_masks[f].detach().cpu().numpy().astype(np.uint8))
                img_E.putpalette(palette)
                img_E.save(os.path.join(anno_save_path_pred, f_id))#"frame_{:04d}.png".format(f_id)))


                if args.overlay_mask :
                


                    frame = f_ims[f].permute(1,2,0).numpy()
                    mx = frame.max()
                    mn = frame.min()
                    frame = (frame - mn)/(mx-mn)
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(origin_w/100,origin_h/100)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.axis('off')
                    fig.add_axes(ax)
                    ax.imshow(frame, alpha = 1, aspect='auto')
                    contour, hier = cv2.findContours(vid_masks[f].numpy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    cmax = None
                    for c in contour:
                        if cmax is None:
                            cmax = c
                        if len(c) > len(cmax):
                            cmax = c

                    if cmax is None:
                        print("cmax is none")
                        
                    else:

                        for c in contour :
                            polygon = Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=colors[0],
                                edgecolor='r', linewidth=0.0,
                                alpha=0.6)
                            ax.add_patch(polygon)
                    

                    fig.savefig(os.path.join(anno_save_path_pred_orig, f_id))#"frame_{:05d}.png".format(f_id)))
                    plt.close(fig)

        

    
    


    return 

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path =  (f"{args.output_dir}/{name_prefix}")
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

    
    backbone = build_vdiff_lean_updown()#VDiffFeatExtractor()    
    model = Vdiff_updown(backbone)
    model.to(args.gpu)
    #model = DDP(model_wo_ddp, device_ids=[args.gpu])#, find_unused_parameters=True)

    
    n_parameters = sum(p.numel() for p in model.backbone.unet.parameters() if p.requires_grad)
    print('number of trainable params unet:', n_parameters)

    

    
    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.backbone.unet.load_state_dict(checkpoint['unet'])
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded ", resume_epoch)
    else:
        print("no checkpoint---------")
        resume_epoch = -999





    evaluate(args, model)


    
    return





if __name__ == "__main__" :

    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    args.log_path, args.val_vis = set_path(args)
    
    main(args)



    