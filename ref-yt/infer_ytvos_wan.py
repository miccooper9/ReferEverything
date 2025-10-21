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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T


import opts


from models.want2v_wrapper import  build_want2vlarge_txt_model, WanVdiff_wrapper



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



transform384 = T.Compose([
    T.Resize((384,384)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])








def evaluate(args, model) :




    split = args.split
    # save path

    save_path_prefix = os.path.join(args.val_vis, "Annotations")
    vis_path_prefix = os.path.join(args.val_vis, "Visualisation")
    

    print(save_path_prefix, vis_path_prefix)
    
    

    # load data
    root = Path(args.ytvos_path)  # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    print(len(valid_test_videos))
    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    print(len(test_videos))
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    print(len(video_list))
    print(video_list)

    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    

    # get palette
    palette_img_path = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette_img = Image.open(palette_img_path)
    palette = palette_img.getpalette()

    max_token_len = -1

    #with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
    
    with torch.no_grad() :
    
        for video in tqdm(video_list):
            metas = []  # list[dict], length is number of expressions

            expressions = data[video]["expressions"]
            expression_list = list(expressions.keys())
            num_expressions = len(expression_list)
            video_len = len(data[video]["frames"])

            print(f"-----{video}-----{num_expressions}")

            # read all the anno meta
            for i in range(num_expressions):
                meta = {}
                meta["video"] = video
                meta["exp"] = expressions[expression_list[i]]["exp"]
                meta["exp_id"] = expression_list[i]
                meta["frames"] = data[video]["frames"]
                metas.append(meta)
            meta = metas

            for i in range(num_expressions):

                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]

                video_len = len(frames)
                print(f"{exp} video_len {video_len}")
                # NOTE: the im2col_step for MSDeformAttention is set as 64
                # so the max length for a clip is 64
                # store the video pred results
                all_pred_logits = []
                all_pred_masks = []

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

                prompt_embeds = prompt_embeds.to(args.gpu, model.backbone.dit.dtype)

                                
                print("prompt embeds : ", prompt_embeds.shape, prompt_embeds.dtype, prompt_embeds.device)

                                


                                
                                
                num_clip_frames = args.num_frames
                                #prompt_in,_ = txtSA(prompt_embeds, mask=masks)
                                #print("prompt_in : ", prompt_in.shape)

                print(list(range(0, video_len, 19)))
                print(list(range(0, video_len, 8)))


                fnums = [73,61,49,37,25,17,9,5]
                fnumidx = fnums.index(num_clip_frames)

                while video_len < num_clip_frames :
                    
                    fnumidx+=1
                    num_clip_frames = fnums[fnumidx]


                print("video len : ", video_len, " num_clip_frames : ", num_clip_frames, " argsF : ", args.num_frames)


                clip_list = [frames[clip_i:clip_i + num_clip_frames] for clip_i in range(0, video_len, num_clip_frames)]
                print("clip lens : ", [len(c) for c in clip_list])
                                
                    
                    
                #track_res = model.generate_empty_tracks()
                vid_masks = []
                f_ims = []
                last_flag = False
                for clip in clip_list:
                    cframes = clip
                    clip_len = len(cframes)
                    cog = clip_len
                    print("======clip len : ", clip_len)
                    print(cframes)

                    if clip_len < num_clip_frames :
                        last_flag =  True
                        cframes = frames[-num_clip_frames : ]
                        clip_len = num_clip_frames
                        print("2======clip len : ", clip_len, cog)
                        print(cframes)

                                        
                                    
                    # store images
                    imgs = []
                    for t in range(clip_len):
                        frame = cframes[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        print(img_path,"--------", np.array(img).shape, origin_w, origin_h)
                        imgs.append(transform(img))
                                    
                    imgs = torch.stack(imgs, dim=0).to(device = args.gpu, dtype=torch.bfloat16)
                    print(f"{exp} img shape {imgs.shape} {video_len}")


                    imgs = imgs.to(args.gpu,model.backbone.vae.dtype)

                    img_latents = model.backbone.vae.encode(imgs.unsqueeze(0).permute(0,2,1,3,4)).latent_dist.mode()
                    latents_mean = (torch.tensor(model.backbone.vae.config.latents_mean).view(1, model.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.backbone.vae.dtype))
                    latents_std = 1.0 / torch.tensor(model.backbone.vae.config.latents_std).view(1, model.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.backbone.vae.dtype)
                    img_latents = (img_latents - latents_mean) * latents_std

                    print("img_latents : ", img_latents.shape, img_latents.max(), img_latents.min())
                    print("p embeds : ", prompt_embeds.shape)


                    pred_latents = model(img_latents.to(model.backbone.dit.dtype), timesteps=0, prompt_embeds=prompt_embeds)



                    pred_latents = pred_latents.to(model.backbone.vae.dtype) / latents_std + latents_mean
                    seg_masks = model.backbone.vae.decode(pred_latents, return_dict=False)[0] #1, 3, t ,h, w

                    print("3===", last_flag)
                    if last_flag :
                        seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)[-cog : , ...]
                        imgs = imgs[-cog : , ...]
                    else :
                        seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)



                                
                    print("seg_masks : ", seg_masks.shape, seg_masks.max(), seg_masks.min())

                    pred_masks = F.interpolate(seg_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)# t, c, h , w
                    rimgs = F.interpolate(imgs, size=(origin_h, origin_w), mode='bilinear',
                                                align_corners=False)
                    
                    pred = pred_masks.mean(1)
                    #pred[pred>0.5] = 1.0
                    #pred[pred<=0.5] = 0.0
                    pred[pred>0.5] = 1.0
                    pred[pred<=0.5] = 0.0
                    print("---------pred :", pred.shape, pred.max(), pred.min())
                    
                    vid_masks.append(pred)
                    f_ims.append(rimgs)


                vid_masks = torch.cat(vid_masks, dim=0).cpu().float()
                f_ims = torch.cat(f_ims, dim=0).cpu().float()
                print(video_len,"-------->>>>", vid_masks.shape, f_ims.shape)

                                

                anno_save_path_pred = os.path.join(save_path_prefix, video_name, exp_id)
                anno_save_path_vis = os.path.join(vis_path_prefix, video_name, exp_id)
                print("anno_save_path_pred : ",anno_save_path_pred)
                print("anno_save_path_pred_orig : ",anno_save_path_vis)

                
                    
                if not os.path.exists(anno_save_path_pred):
                    os.makedirs(anno_save_path_pred)
                if not os.path.exists(anno_save_path_vis):
                    os.makedirs(anno_save_path_vis)


                for f in tqdm(range(vid_masks.shape[0])):
                    #fig, ax = plt.subplots(2, 1, figsize=(20, 30))
                    print("<<<>>>>", len(frames))
                    frame_name = frames[f]


                    mask = vid_masks[f].detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(anno_save_path_pred, frame_name + ".png")
                    mask.save(save_file)


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
                    ax.imshow(vid_masks[f].numpy(), alpha = 0.5, interpolation= 'none')
                    fig.savefig(os.path.join(anno_save_path_vis, f'{frame_name}.png'))
                    plt.close(fig)
                    

                                    

                                        
                                    


    
    


    return












                





                






        


        
        


            

            
    

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path = (f"{args.output_dir}/{name_prefix}_f{args.num_frames}")

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



