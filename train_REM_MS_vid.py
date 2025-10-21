
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
from einops import rearrange

#import datasets.samplers as samplers
from datasets import build_dataset
from davis2017.evaluation import DAVISEvaluation
#from engine import train_one_epoch, evaluate, evaluate_a2d, evaluate_online_a2d
#from models import build_model
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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

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

def evaluate(args, model, epoch) :

    root = Path(args.davis_path)  # data/ref-davis
    img_folder = os.path.join(root, 'valid', "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", 'valid', "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    save_path_prefix = os.path.join(args.val_vis, f'r_davis_{epoch}')

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

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
            metas.append(meta)


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

                    if max_token_len <= len(tokens) :
                        max_token_len = len(tokens)


                    prompt_embeds, masks, attention_mask = prompt_embeds[:,:args.token_length,:].to(args.device), masks.to(args.device), attention_mask.to(args.device)
                    
                    
                    num_clip_frames = args.num_frames
                    #prompt_in,_ = txtSA(prompt_embeds, mask=masks)
                    #print("prompt_in : ", prompt_in.shape)
                    
                    
                    
                    # 3. for each clip
                    vid_masks = []
                    for clip_id in range(0, video_len, num_clip_frames):
                        frames_ids = [x for x in range(video_len)]
                        clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                        #print(clip_frames_ids)
                        clip_len = len(clip_frames_ids)


                        
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
                                tim = transform256(img).to(args.device)  # list[Img]
                            else :
                                tim = transform(img).to(args.device)  # list[Img]
                            pframes.append(tim)
                            
                        imgs = torch.stack(pframes, dim=0).to(args.device)
                        print(f"{exp} img shape {imgs.shape} {video_len}")


                        pred_latents = model(imgs.unsqueeze(0),prompt_embeds)
                        
                        p_latents = 1 / model.backbone.vae.config.scaling_factor * pred_latents.detach()
                        print("unscaled pred latents : ", p_latents.shape, p_latents.max(), p_latents.min())
                        seg_masks = model.backbone.vae.decode(p_latents).sample
                        
                        print("seg_masks : ", seg_masks.shape, seg_masks.max(), seg_masks.min())

                        
                        pred_masks = F.interpolate(seg_masks, size=(origin_h, origin_w), mode='bilinear',
                                                align_corners=False)# t, c, h , w
                    
                        pred = pred_masks.mean(1)
                        pred[pred>0.5] = 1.0
                        pred[pred<=0.5] = 0.0
                        pred = pred*(obj_id+1)
                        print("---------pred :", pred.shape, pred.max(), pred.min())
                        vid_masks.append(pred)


                    vid_masks = torch.cat(vid_masks, dim=0)
                    print(video_len,"-------->>>>", vid_masks.shape)
                    anno_masks.append(vid_masks)

                    anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video, str(obj_id+1))
                    print("anno_save_path : ",anno_save_path)
                    if not os.path.exists(anno_save_path):
                        os.makedirs(anno_save_path)
                    for f in range(vid_masks.shape[0]):
                        img_E = Image.fromarray(vid_masks[f].detach().cpu().numpy().astype(np.uint8))
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))

    
    print("Calculating metrics ....")
    sum_JFmean = 0
    for anno_id in range(4):
    
        dataset_eval = DAVISEvaluation(davis_root='/mnt/fsx-west/anurag/r-davis/DAVIS', task='unsupervised', gt_set='val')
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
    print(f'{epoch} result : {JFm}')
    wandb.log({"R-Davis J&F":JFm, 'Steps': epoch})


    return JFm







def vis_train(savepath,pred,target,re_target,image,iter) :

    samples = random.sample(range(target.shape[0]), 1)

    pred = pred[samples]
    target = target[samples]
    image = image[samples]
    re_target = re_target[samples]

    print(pred.shape)

    _,nf,_,_ = pred.shape

    fig, ax = plt.subplots(4, nf, figsize=(35, 20))
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
            seg_mask[seg_mask > 0.5] = 1.0
            seg_mask[seg_mask <= 0.5] = 0.0
            print("inside train vis")
            print(pred[i,f,...].max(), pred[i,f,...].min(), seg_mask.max(), seg_mask.min())
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
        j= j+4

    fig.savefig(f'{savepath}/vis_{iter}.png')
    plt.close(fig)



def train_one_epoch(args, model, loader, optimizer):

    

    


    for idx,batch in enumerate(tqdm(loader)):

        

        img, target_dict = batch


        target = target_dict['masks'] # b t c h w
        img = img # b t c h w

        prompts = target_dict['caption']
        print(prompts)

        

        text_in = model.module.backbone.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.module.backbone.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")

        with torch.no_grad() :
            prompt_embeds = model.module.backbone.text_encoder(
                            text_in.input_ids.to(args.gpu),#attention_mask=text_in.attention_mask.to(args.device),
                            )[0]
        
        print(prompt_embeds.shape,prompt_embeds.device)
        
        attention_mask = text_in['attention_mask'][:, :args.token_length]
        print(args.gpu, ": ","attn mask : ", attention_mask[0,0:20])
        masks = ~attention_mask.to(torch.bool)
        print(args.gpu, ": ","mask : ", masks[0,0:20])

        

        img, target, prompt_embeds, masks = img.to(args.gpu), target.float().to(args.gpu), prompt_embeds[:,:args.token_length,:].to(args.gpu), masks.to(args.gpu)

        print(args.gpu, ": ","-------------",img.shape, target.shape, prompt_embeds.shape, masks.shape)
        print(args.gpu, ": ","-------------",img.device, target.device, prompt_embeds.device, masks.device)

        b,nf,_,_,_ = img.shape

        with torch.no_grad() :

            target_r = target.unsqueeze(2).repeat(1,1,3,1,1).squeeze(0)
            #target_r = rearrange(target_r, 'b f c h w -> (b f) c h w')
            print(args.gpu, ": ", "target_r : ", target_r.shape, target_r.max(), target_r.min())
            target_latents = model.module.backbone.vae.encode(target_r).latent_dist.mode() * model.module.backbone.vae.config.scaling_factor


        print(args.gpu, ": ", "target_latent : ", target_latents.shape, target_latents.max(), target_latents.min())
        
        pred_latents = model(img,prompt_embeds)

        print(args.gpu, ": ", "pred_latent : ", pred_latents.shape, pred_latents.max(), pred_latents.min())

        mse = nn.MSELoss()
        loss = mse(pred_latents, target_latents)
        
        print(args.gpu, ": ", "loss : ", loss)
        optimizer.zero_grad()  
        loss.backward()
        

        optimizer.step()

        if args.gpu == 0 :

            wandb.log({"YTVOS loss":loss.detach(), 'Steps': args.iteration})

        args.iteration+=1

        if args.gpu == 0 and args.iteration%500 == 0 :

            p_latents = 1 / model.module.backbone.vae.config.scaling_factor * pred_latents.detach()
            print("unscaled pred latents : ", p_latents.shape, p_latents.max(), p_latents.min())
            pred = model.module.backbone.vae.decode(p_latents).sample
            print("pred :", pred.shape, pred.max(), pred.min())
            pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b,t=nf).mean(2)
            print("pred : ", pred.shape)

            t_latents = 1 / model.module.backbone.vae.config.scaling_factor * target_latents.detach()
            print("unscaled target latents : ", t_latents.shape, t_latents.max(), t_latents.min())
            re_target = model.module.backbone.vae.decode(t_latents).sample
            print("re_target :", re_target.shape, re_target.max(), re_target.min())
            re_target = rearrange(re_target, '(b t) c h w -> b t c h w', b=b,t=nf).mean(2)
            print("re_target : ", re_target.shape)

            vis_train(args.train_vis,pred.detach().cpu(),target.detach().cpu(),re_target.detach().cpu(),img.detach().cpu(),args.iteration)
        
        
    



        
        


            

            
    

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
            group = 'R-VOS',
            name= f'{args.name_prefix}_lr{args.lr}_f{args.num_frames}'
        )
        wandb.define_metric("Steps")
        wandb.define_metric("*", step_metric="Steps")
    



    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, num_workers=16,shuffle=False, drop_last=True, sampler = DistributedSampler(dataset_train))

    
    backbone = build_vdiff_lean_updown()#VDiffFeatExtractor()    
    model_wo_ddp = Vdiff_updown(backbone)
    model_wo_ddp.to(args.gpu)
    model = DDP(model_wo_ddp, device_ids=[args.gpu])#, find_unused_parameters=True)

    
    n_parameters = sum(p.numel() for p in model.module.backbone.unet.parameters() if p.requires_grad)
    print('number of trainable params unet:', n_parameters)

    

    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale', 'norm']
    param_group_no_decay = []
    param_group_with_decay = []
    unet_param_group_no_decay = []
    unet_param_group_with_decay = []

    for name, param in model.module.backbone.unet.named_parameters():
        if not param.requires_grad:
            print(f'Param not requires_grad: {name}')
            continue
        if any([i in name for i in no_decay]):
            print(name)
            unet_param_group_no_decay.append(param)
        else:
            unet_param_group_with_decay.append(param)

        param.register_hook(lambda grad: torch.clamp(grad, -args.clip_val, args.clip_val))


    

    params = []

    params.append({'params': unet_param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
    params.append({'params': unet_param_group_with_decay, 'lr': args.lr, 'weight_decay': args.weight_decay})

    optimizer = torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay,amsgrad=args.amsgrad)

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        model.module.backbone.unet.load_state_dict(checkpoint['unet'])
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded ", resume_epoch)
        
    else:
        print("no checkpoint---------")
        resume_epoch = -999


    args.iteration = -1
    bestJFm = -1


    for epoch in range(max(0, resume_epoch+1), args.epochs):

        train_one_epoch(args, model, train_loader, optimizer)

        if args.gpu == 0 :


            JFm = evaluate(args, model.module, epoch)

            if bestJFm < JFm :

                bestJFm = JFm

                dict_to_save = {
                                'unet': model.module.backbone.unet.state_dict(),
                                'optimizer': optimizer.state_dict(), 
                                'epoch': epoch, 
                                'args': args,
                                }

                print(dict_to_save.keys())

                torch.save(dict_to_save, os.path.join(args.model_path,f'model_ckpt{epoch}.pth'))

        dist.barrier()

        


        


        


        


        


        
        

    dist.destroy_process_group()
    
    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    
    

    main(args)



