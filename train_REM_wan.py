
import argparse
import datetime
import json
import random

from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from PIL import Image

#import datasets.samplers as samplers
from datasets import build_dataset
from davis2017.evaluation import DAVISEvaluation

from tqdm import tqdm
import os
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
import wandb
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.models.transformers.transformer_wan import WanTransformerBlock, WanTimeTextImageEmbedding
from diffusers.models.autoencoders.autoencoder_kl_wan import WanEncoder3d, WanDecoder3d
from diffusers.models.normalization import FP32LayerNorm, RMSNorm


from models.want2v_wrapper import  build_want2v_txtfree_model, WanVdiff_wrapper
import opts
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    CustomPolicy,
    enable_wrap,
    wrap,
)

from transformers import UMT5EncoderModel
import gc
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

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


transformwan = T.Compose([
    T.Resize((480,832)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])






def apply_fsdp_checkpointing(model, block, p):
    """
    Apply selective activation checkpointing.

    Selectivity is defined as a percentage p, which means we apply ac
    on p of the total blocks. p is a floating number in the range of
    [0, 1].

    Some examples:
    p = 0: no ac for all blocks. same as `fsdp_activation_checkpointing=False`
    p = 1: apply ac on every block. i.e. "full ac".
    p = 1/2: [ac, no-ac, ac, no-ac, ...]
    p = 1/3: [no-ac, ac, no-ac,   no-ac, ac, no-ac,   ...]
    p = 2/3: [ac, no-ac, ac,    ac, no-ac, ac,    ...]
    Since blocks are homogeneous, we make ac blocks evenly spaced among
    all blocks.

    Implementation:
    For a given ac ratio p, we should essentially apply ac on every "1/p"
    blocks. The first ac block can be as early as the 0th block, or as
    late as the "1/p"th block, and we pick the middle one: (0.5p)th block.
    Therefore, we are essentially to apply ac on:
    (0.5/p)th block, (1.5/p)th block, (2.5/p)th block, etc., and of course,
    with these values rounding to integers.
    Since ac is applied recursively, we can simply use the following math
    in the code to apply ac on corresponding blocks.
    """
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        #print("======the submodule : ", submodule)
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, block):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )





def evaluate_iter(args, model, epoch) :

    #hack to get around torch fsdp bug
    with torch.no_grad() :
        _ = model(
            torch.randn(1, 16, 3, 32, 32).to(args.gpu), 
            torch.randn(1, 77, 4096).to(args.gpu), 
            timesteps=0)

    root = Path(args.davis_path)  # data/ref-davis
    img_folder = os.path.join(root, 'valid', "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", 'valid', "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    save_path_prefix = os.path.join(args.val_vis, f'r_davis_{args.iteration}')

    if args.gpu == 0 :
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)

    dist.barrier()

    # get palette
    palette_img_path = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette_img = Image.open(palette_img_path)
    palette = palette_img.getpalette()

    max_token_len = -1
    
    for video in tqdm(video_list):

        metas = []

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
            meta["exp_id"] = expression_list[i]  # start from 0
            meta["frames"] = data[video]["frames"]
            meta["oid"] = expressions[expression_list[i]]["obj_id"]
            metas.append(meta)

            print(i,"-==========----<<<<<<<",f'{meta["video"]}_{meta["exp_id"]}_{meta["oid"]}')


            exp=meta["exp"]
            #print(f"----------{exp}")


        meta = metas
        # since there are 4 annotations
        num_obj = num_expressions // 4


        for anno_id in range(4):  # 4 annotators
            anno_logits = []
            anno_masks = []  # [num_obj+1, video_len, h, w], +1 for background

            with torch.no_grad() :
                for obj_id in range(num_obj):
                    i = obj_id * 4 + anno_id
                    video_name = meta[i]["video"]
                    exp = meta[i]["exp"]
                    exp_id = meta[i]["exp_id"]
                    frames = meta[i]["frames"]
                    oid = meta[i]["oid"]

                    video_len = len(frames)
                    print(f"{exp} video_len {video_len}")
                    # NOTE: the im2col_step for MSDeformAttention is set as 64
                    # so the max length for a clip is 64
                    # store the video pred results
                    all_pred_logits = []
                    all_pred_masks = []

                    

                    p_emb_pth = os.path.join(args.T5_enc_path, 'rdavis', f'{video}_{exp_id}_{oid}_nonull_temb.pt')
                    prompt_embeds = torch.load(p_emb_pth, map_location="cpu")
                    prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(args.token_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)
                    prompt_embeds = prompt_embeds.to(args.gpu, model.module.backbone.dit.dtype)

                    print("prompt_embeds : ", prompt_embeds.shape, prompt_embeds.dtype )
                    
                    
                    num_clip_frames = args.num_frames

                    
                    
                    
                    # 3. for each clip
                    vid_masks = []
                    last_flag = False
                    for clip_id in range(0, video_len, num_clip_frames):
                        frames_ids = [x for x in range(video_len)]
                        clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                        #print(clip_frames_ids)
                        clip_len = len(clip_frames_ids)

                        if clip_len < num_clip_frames :
                            last_flag =  True
                            clip_frames_ids = frames_ids[-num_clip_frames : ]


                        
                        # load the clip images
                        pframes = []
                        for t in clip_frames_ids:
                            frame = frames[t]
                            img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                            img = Image.open(img_path).convert('RGB')
                            origin_w, origin_h = img.size
                            print(img_path,"--------", np.array(img).shape, origin_w, origin_h)
                            #print(palette_img_path, np.array(palette_img).shape)
                            if args.half :
                                tim = transform256(img).to(args.gpu)  # list[Img]
                            elif args.wanres :
                                tim = transformwan(img).to(args.gpu)
                            else :
                                tim = transform(img).to(args.gpu)  # list[Img]
                            pframes.append(tim)
                            
                        imgs = torch.stack(pframes, dim=0).to(args.gpu)
                        print(f"{exp} img shape {imgs.shape} {video_len}")


                        imgs = imgs.to(args.gpu,model.module.backbone.vae.dtype)

                        img_latents = model.module.backbone.vae.encode(imgs.unsqueeze(0).permute(0,2,1,3,4)).latent_dist.mode()
                        latents_mean = (torch.tensor(model.module.backbone.vae.config.latents_mean).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.module.backbone.vae.dtype))
                        latents_std = 1.0 / torch.tensor(model.module.backbone.vae.config.latents_std).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.module.backbone.vae.dtype)
                        img_latents = (img_latents - latents_mean) * latents_std

                        print("img_latents : ", img_latents.shape, img_latents.max(), img_latents.min())
                        print("p embeds : ", prompt_embeds.shape)

                        pred_latents = model(img_latents.to(model.module.backbone.dit.dtype), timesteps=0, prompt_embeds=prompt_embeds)

                        pred_latents = pred_latents.to(model.module.backbone.vae.dtype) / latents_std + latents_mean
                        seg_masks = model.module.backbone.vae.decode(pred_latents, return_dict=False)[0] #1, 3, t ,h, w

                        if last_flag :
                            seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)[-clip_len : , ...]
                        else :
                            seg_masks = seg_masks.squeeze(0).permute(1,0,2,3)
                        
                        print("seg_masks : ", seg_masks.shape, seg_masks.max(), seg_masks.min()) 


                        
                        pred_masks = F.interpolate(seg_masks, size=(origin_h, origin_w), mode='bilinear',
                                                align_corners=False)# t, c, h , w
                    
                        pred = pred_masks.mean(1)
                        pred[pred>0.5] = 1.0
                        pred[pred<=0.5] = 0.0
                        pred = pred*(obj_id+1)
                        print("---------pred :", pred.shape, pred.max(), pred.min())
                        vid_masks.append(pred.float())


                    vid_masks = torch.cat(vid_masks, dim=0)
                    print(video_len,"-------->>>>", vid_masks.shape)
                    anno_masks.append(vid_masks)

                    if args.gpu == 0 :

                        anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video, str(obj_id+1))
                        print("anno_save_path : ",anno_save_path)
                        if not os.path.exists(anno_save_path):
                            os.makedirs(anno_save_path)
                        for f in range(vid_masks.shape[0]):
                            img_E = Image.fromarray(vid_masks[f].detach().cpu().numpy().astype(np.uint8))
                            img_E.putpalette(palette)
                            img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))

                    dist.barrier()

    
    print("Calculating metrics ....")
    sum_JFmean = 0
    for anno_id in range(4):
    
        dataset_eval = DAVISEvaluation(davis_root=args.davis_anno_path, task='unsupervised', gt_set='val')
        results_path = os.path.join(save_path_prefix, f"anno_{anno_id}")
        metrics_res = dataset_eval.evaluate(results_path)

        J_, F_ = metrics_res['J'], metrics_res['F']


        csv_name_global = f'global_results-unsupervised-val.csv'
        csv_name_per_sequence = f'per-sequence_results-unsupervised-val.csv'
        csv_name_global_path = os.path.join(results_path, csv_name_global)
        csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)

        # Generate dataframe for the general results
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J_["M"]) + np.mean(F_["M"])) / 2.
        sum_JFmean += final_mean

        g_res = np.array([final_mean, np.mean(J_["M"]), np.mean(J_["R"]), np.mean(J_["D"]), np.mean(F_["M"]), np.mean(F_["R"]),
                        np.mean(F_["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.5f")
        print(f'Global results saved in {csv_name_global_path}')

        # Generate a dataframe for the per sequence results
        seq_names = list(J_['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J_['M_per_object'][x] for x in seq_names]
        F_per_object = [F_['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.5f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

        # Print the results
        print(f"--------------------------- Global results for val ---------------------------\n")
        print(table_g.to_string(index=False))
        print(f"\n---------- Per sequence results for val ----------\n")
        print(table_seq.to_string(index=False))

    JFm = sum_JFmean/4
    if args.gpu == 0 : 
        print(f'{args.iteration} result : {JFm}')
        wandb.log({"R-Davis J&F":JFm, 'Steps': args.iteration})

    dist.barrier()


    return JFm








def vis_train(savepath,pred,target,re_target,image,iter) :

    samples = random.sample(range(target.shape[0]), 1)

    pred = pred[samples]
    target = target[samples]
    image = image[samples]
    re_target = re_target[samples]

    print("vis train", pred.shape)

    _,nf,_,_ = pred.shape

    fig, ax = plt.subplots(5, nf, figsize=(35, 25))
    j=-1
    for i in range(1) :
        

        for f in range(nf) :

            print("---------->", image.shape)

            frame = image[i,f,...].permute(1,2,0).numpy()
            mx = frame.max()
            mn = frame.min()
            frame = (frame - mn)/(mx-mn)
            gt_mask = target[i,f,...].numpy()
            re_gt_mask = re_target[i,f,...].numpy()
            re_gt_mask[re_gt_mask > 0.5] = 1.0
            re_gt_mask[re_gt_mask <= 0.5] = 0.0
            seg_mask = pred[i,f,...].numpy()

            seg_max = seg_mask.max()
            seg_min = seg_mask.min()
            seg_image = (seg_mask - seg_min)/(seg_max-seg_min)

            seg_mask[seg_mask > 0.5] = 1.0
            seg_mask[seg_mask <= 0.5] = 0.0
            print("inside train vis")
            print(pred[i,f,...].max(), pred[i,f,...].min(), seg_mask.max(), seg_mask.min())

            if nf==1 :
                ax[j+1].imshow(frame)
                ax[j+1].set_title(f'image-{i}')
                ax[j+2].imshow(frame)
                ax[j+2].imshow(gt_mask, alpha = 0.5, interpolation= 'none')
                ax[j+2].set_title(f'GT mask-{i}')
                ax[j+3].imshow(frame)
                ax[j+3].imshow(re_gt_mask, alpha = 0.5, interpolation= 'none')
                ax[j+3].set_title(f'Recon GT mask-{i}')
                ax[j+4].imshow(frame)
                ax[j+4].imshow(seg_mask, alpha = 0.5, interpolation= 'none')
                ax[j+4].set_title(f'Seg mask-{i}')
                ax[j+5].imshow(seg_image)
                ax[j+5].set_title(f'seg_image-{i}')

            else :
                ax[j+1,f].imshow(frame)
                ax[j+1,f].set_title(f'image-{i}')
                ax[j+2,f].imshow(frame)
                ax[j+2,f].imshow(gt_mask, alpha = 0.5, interpolation= 'none')
                ax[j+2,f].set_title(f'GT mask-{i}')
                ax[j+3,f].imshow(frame)
                ax[j+3,f].imshow(re_gt_mask, alpha = 0.5, interpolation= 'none')
                ax[j+3,f].set_title(f'Recon GT mask-{i}')
                ax[j+4,f].imshow(frame)
                ax[j+4,f].imshow(seg_mask, alpha = 0.5, interpolation= 'none')
                ax[j+4,f].set_title(f'Seg mask-{i}')
                ax[j+5,f].imshow(seg_image)
                ax[j+5,f].set_title(f'seg_image-{i}')
        j= j+5

    fig.savefig(f'{savepath}/vis_{iter}.png')
    plt.close(fig)



def train_one_epoch(args, model, im_loader, vid_loader, optimizer, scheduler, epoch):

    #hack to get around torch fsdp bug
    with torch.no_grad() :
        _ = model(
            torch.randn(1, 16, 3, 32, 32).to(device = args.gpu, dtype=torch.bfloat16), 
            torch.randn(1, 77, 4096).to(device = args.gpu, dtype=torch.bfloat16), 
            timesteps=0)



    v_data = iter(vid_loader)
    i_data = iter(im_loader)

    for idx in tqdm(range(2*len(im_loader))) :


        if args.iteration%2 == 0 :
            try :
                batch = next(i_data)
            except StopIteration:
                i_data = iter(im_loader)
                batch = next(i_data)

        else :
            try :
                batch = next(v_data)
            except StopIteration:
                v_data = iter(vid_loader)
                batch = next(v_data)
        img, target_dict = batch


        target = target_dict['masks'] # b t c h w
        img = img # b t c h w

        prompts = target_dict['caption']
        print(prompts)
        prompts = [prompt_clean(u) for u in prompts]
        print("post clean ", prompts)

       
        #prompt_embeds = torch.randn(1, 77, 4096)
        prompt_embeds = target_dict['p_emb']#.to(device=args.gpu, dtype=torch.bfloat16)
        prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(args.token_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
            )
        print("prompt_embeds : ", prompt_embeds.shape, prompt_embeds.dtype )
        
        
        


        

        img, target, prompt_embeds = img.to(args.gpu,model.module.backbone.vae.dtype), target.to(args.gpu, model.module.backbone.vae.dtype), prompt_embeds.to(args.gpu, model.module.backbone.dit.dtype)

        print(args.gpu, ": ","-------------",img.shape, img.max(), img.min(), target.shape, target.max(), target.min(), prompt_embeds.shape)
        print(args.gpu, ": ","-------------",img.dtype, img.device, target.dtype, target.device, prompt_embeds.dtype, prompt_embeds.device)

        b,nf,_,_,_ = img.shape

        with torch.no_grad() :


            print("=-=-=-",img.permute(0,2,1,3,4).dtype)
            with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
                img_latents = model.module.backbone.vae.encode(img.permute(0,2,1,3,4)).latent_dist.mode()


            latents_mean = (torch.tensor(model.module.backbone.vae.config.latents_mean).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.module.backbone.vae.dtype))
            latents_std = 1.0 / torch.tensor(model.module.backbone.vae.config.latents_std).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(args.gpu, model.module.backbone.vae.dtype)

            img_latents = (img_latents - latents_mean) * latents_std

            target_r = target.unsqueeze(2).repeat(1,1,3,1,1)

            with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
                target_latents = model.module.backbone.vae.encode(target_r.permute(0,2,1,3,4)).latent_dist.mode()

            target_latents = (target_latents - latents_mean) * latents_std



        print(args.gpu, ": ", "img_latents : ", img_latents.shape, img_latents.max(), img_latents.min())
        print(args.gpu, ": ", "target_latents : ", target_latents.shape, target_latents.max(), target_latents.min())
        print(args.gpu, ": ", "prompt_embeds : ", prompt_embeds.shape, prompt_embeds.dtype)
        
        pred_latents = model(img_latents.to(model.module.backbone.dit.dtype), prompt_embeds, timesteps=0)

        print(args.gpu, ": ", "pred_latent : ", pred_latents.shape, pred_latents.max(), pred_latents.min())

        mse = nn.MSELoss()
        loss = mse(pred_latents, target_latents.to(model.module.backbone.dit.dtype))
        
        print(args.gpu, ": ", "loss : ", loss)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()

        

        optimizer.step()

        
        if args.gpu == 0 :

            wandb.log({"YTVOS loss":loss.detach(), 'Steps': args.iteration})

        scheduler.step()

        if args.gpu == 0 and args.iteration % 100 == 0:
            wandb.log({"LR":scheduler.get_last_lr()[0], 'Steps': args.iteration})
            
        args.iteration+=1

        with torch.no_grad() :

            print("<><><>>>>><<<<<<<<<<<")

            if args.iteration%250 == 0 :


                
                #latents_mean = (torch.tensor(model.module.backbone.vae.config.latents_mean).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(model.module.backbone.vae.device, model.module.backbone.vae.dtype))
                #latents_std = 1.0 / torch.tensor(model.module.backbone.vae.config.latents_std).view(1, model.module.backbone.vae.config.z_dim, 1, 1, 1).to(model.module.backbone.vae.device, model.module.backbone.vae.dtype)

                pred_latents = pred_latents.to(model.module.backbone.vae.dtype) / latents_std + latents_mean

                with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
                    pred = model.module.backbone.vae.decode(pred_latents, return_dict=False)[0].mean(1)

                print("pred :", pred.shape, pred.max(), pred.min())
                

                target_latents = target_latents / latents_std + latents_mean

                with torch.autocast("cuda", torch.bfloat16, cache_enabled=False) :
                    re_target = model.module.backbone.vae.decode(target_latents, return_dict=False)[0].mean(1)
                
                print("re_target :", re_target.shape, re_target.max(), re_target.min())

                if args.gpu==0 :
                

                    vis_train(args.train_vis,pred.detach().cpu().float(),target.detach().cpu().float(),re_target.detach().cpu().float(),img.detach().cpu().float(),args.iteration)

        if args.iteration%5000 == 0 :

            JFm = evaluate_iter(args, model, epoch)

            if args.bestJFm < JFm :

                

                args.bestJFm = JFm

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy) :
                    
                    cpu_state = model.state_dict()
                    


                    for k in cpu_state.keys() :
                        print("c<<<<<",k)


                
                if args.gpu == 0 :

                    
                    dict_to_save = {
                                'model_state': cpu_state,
                                'optimizer': optimizer.state_dict(), 
                                'epoch': epoch,
                                'iteration' : args.iteration, 
                                'args': args,
                                }

                    print(dict_to_save.keys())

                    torch.save(dict_to_save, os.path.join(args.model_path,f'model_ckpt{epoch}_{args.iteration}.pth'))

            else :


                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy) :
                    
                    cpu_state = model.state_dict()
                    


                    for k in cpu_state.keys() :
                        print("c<<<<<",k)


                
                if args.gpu == 0 :

                    
                    dict_to_save = {
                                'model_state': cpu_state,
                                'optimizer': optimizer.state_dict(), 
                                'epoch': epoch,
                                'iteration' : args.iteration, 
                                'args': args,
                                }

                    print(dict_to_save.keys())

                    torch.save(dict_to_save, os.path.join(args.model_path,f'model_ckpt{epoch}_{args.iteration}_latest.pth'))


        dist.barrier()



        
        
    



        
        


            

            
    

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path = (f"{args.output_dir}/{name_prefix}"
        f"bs{args.batch_size}_lr{args.lr}")
    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')
    train_vis = os.path.join(exp_path, 'train_vis')
    val_vis = os.path.join(exp_path, 'val_vis')

    if args.gpu == 0 :
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        if not os.path.exists(train_vis): 
            os.makedirs(train_vis)
        if not os.path.exists(val_vis): 
            os.makedirs(val_vis)
    
        with open(f'{log_path}/running_command.txt', 'a') as f:
            json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
            f.write('\n')

    return log_path, model_path, train_vis, val_vis




def main(args):

    args.gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl',timeout=timedelta(seconds=10000))

    print("device : ", args.gpu)

    args.log_path, args.model_path, args.train_vis, args.val_vis = set_path(args)

    if args.gpu == 0 :

        

        wandb.init(
            # set the wandb project where this run will be logged
            project="ReferSeg-project",

            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,

            },
            #group = 'R-VOS',
            name= f'{args.name_prefix}_lr{args.lr}_f{args.num_frames}'
        )
        wandb.define_metric("Steps")
        wandb.define_metric("*", step_metric="Steps")
    



    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    cogx_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            
            WanTransformerBlock,
            WanTimeTextImageEmbedding,


            WanEncoder3d,
            WanDecoder3d,
            nn.Linear,
            nn.LayerNorm,
            nn.Conv3d,
            FP32LayerNorm,
            RMSNorm


        },
    )

    
    backbone = build_want2v_txtfree_model(args.base_dir)
    model_wo_ddp = WanVdiff_wrapper(backbone)





    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_wo_ddp.load_state_dict(checkpoint['model_state'])
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded ", resume_epoch)
        
    else:
        print("no checkpoint---------")
        resume_epoch = -999


    n_parameters = sum(p.numel() for p in model_wo_ddp.backbone.parameters() if p.requires_grad)
    print('number of trainable params pre fsdp:', n_parameters)
    


    

    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)

    
    bfSixteen = MixedPrecision(
                param_dtype=torch.bfloat16,#torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.float32,#torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.float32,
            )
    

    

    

    

    # Step 3: wrap rest of model
    model = FSDP(
        model_wo_ddp,
        auto_wrap_policy=cogx_auto_wrap_policy,#custom_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=args.gpu,
    )



    apply_fsdp_checkpointing(model, Attention, 0.99)# , 0.95)
    apply_fsdp_checkpointing(model, FeedForward, 0.99)
    
    




    
    

    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale', 'norm']
    param_group_no_decay = []
    param_group_with_decay = []
    
    for name, param in model.named_parameters() :
        #print(param.device)
        if not param.requires_grad:
            continue
        print("paran requires grad",name)

        if any([i in name for i in no_decay]):
            print("no decay : ", name)
            param_group_no_decay.append(param)
        else:
            
            param_group_with_decay.append(param)

        #param.register_hook(lambda grad: torch.clamp(grad, -1e-1, 1e-1))


    params = []
    params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.weight_decay})
    
    def lr_lambda(step):
        return 1.0 if step < 3500 else 0.1  # 1e-5 * 0.1 = 1e-6

    optimizer = torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay,amsgrad=False)

    scheduler = LambdaLR(optimizer, lr_lambda)


    fn = args.num_frames
    args.num_frames = 1
    dataset_im = build_dataset("joint_im_T5", image_set='train', args=args)
    im_loader = torch.utils.data.DataLoader(dataset_im, batch_size=1, num_workers=16,shuffle=False, drop_last=True, sampler = DistributedSampler(dataset_im))
    print("im_loader : ",len(im_loader))

    args.num_frames = fn
    dataset_vid = build_dataset("ytvos_T5", image_set='train', args=args)
    vid_loader = torch.utils.data.DataLoader(dataset_vid, batch_size=1, num_workers=16,shuffle=False, drop_last=True, sampler = DistributedSampler(dataset_vid))
    print("vid_loader : ",len(vid_loader))

    n_parameters = sum(p.numel() for p in model.module.backbone.dit.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)

    args.iteration = 0
    args.bestJFm = -1


    for epoch in range(0, args.epochs):



        train_one_epoch(args, model, im_loader, vid_loader, optimizer, scheduler, epoch)

       

    dist.destroy_process_group()
    
    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    
    

    main(args)



