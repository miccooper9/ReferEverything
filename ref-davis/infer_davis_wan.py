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

from davis2017.evaluation import DAVISEvaluation

from tqdm import tqdm
import os
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd




from models.want2v_wrapper import  build_want2vlarge_txt_model, WanVdiff_wrapper
import opts



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

                    #print("prompt embeds : ", prompt_embeds.shape)
                    #for k,v in text_in.items() :
                    #    print(k,"----", v.shape)

                    #prompt_embeds = torch.randn(1, 77, 4096).to(args.gpu, model.backbone.dit.dtype)

                    #p_emb_pth = os.path.join(args.T5_enc_path, 'rdavis', f'{video}_{exp_id}_{oid}_nonull_temb.pt')
                    #prompt_embeds = torch.load(p_emb_pth, map_location="cpu")
                    #prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(args.token_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)
                    prompt_embeds = prompt_embeds.to(args.gpu, model.backbone.dit.dtype)

                    print("prompt_embeds : ", prompt_embeds.shape, prompt_embeds.dtype )
                    
                    
                    num_clip_frames = args.num_frames
                    #prompt_in,_ = txtSA(prompt_embeds, mask=masks)
                    #print("prompt_in : ", prompt_in.shape)
                    fnums = [73,61,49,37,25,17,9]
                    fnumidx = fnums.index(num_clip_frames)

                    while video_len < num_clip_frames :
                        
                        fnumidx+=1
                        num_clip_frames = fnums[fnumidx]

                    print("video len : ", video_len, " num_clip_frames : ", num_clip_frames, " argsF : ", args.num_frames)
                    
                    
                    
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

    print(f'F{args.num_frames} {epoch} result : {JFm}')


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






        
        
def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.output_dir}/{args.name_prefix}_{args.num_frames}"
    exp_path = (f"{name_prefix}")
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
        JFm = evaluate(args, model, resume_epoch)


    return





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    
    for f in [73] : #73
        args.num_frames = f
        args.log_path, args.val_vis = set_path(args)
        main(args)



