import sys
sys.path.insert(1, "../")

import argparse
import json
import random

from pathlib import Path
from datetime import datetime
import numpy as np
import torch



import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

from PIL import Image
import torch.nn.functional as F
import json


from models.mst2v_wrapper import Vdiff_updown, build_vdiff_lean_updown
import opts
from tqdm import tqdm





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

    

    print("single gpu eval")

    single_processor(args, data, save_path_prefix, vis_path_prefix, img_folder, video_list)

    

def single_processor(args, data, save_path_prefix, vis_path_prefix, img_folder, video_list):


    backbone = build_vdiff_lean_updown()#VDiffFeatExtractor()    
    model = Vdiff_updown(backbone)
    model.to(args.gpu)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.backbone.unet.load_state_dict(checkpoint['unet'])
        resume_epoch = checkpoint['epoch']
        print(f"checkpoint loaded : {resume_epoch}")
    else:
        print("no checkpoint---------")
        resume_epoch = -999

    
    # start inference
    
    # 1. For each video

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

        # 2. For each expression
        with torch.no_grad() :
            for i in range(num_expressions):
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]

                video_len = len(frames)
                print(f"{exp} video_len {video_len}")


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
                print("prompt embeds : ", prompt_embeds.shape)
                
                num_clip_frames = args.num_frames
            
                
                clip_list = [frames[clip_i:clip_i + num_clip_frames] for clip_i in range(0, video_len, num_clip_frames)]
                print("clip lens : ", [len(c) for c in clip_list])

                #track_res = model.generate_empty_tracks()
                vid_masks = []
                f_ims = []
                for clip in clip_list:
                    cframes = clip
                    clip_len = len(cframes)

                    # store images
                    imgs = []
                    for t in range(clip_len):
                        frame = cframes[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        print(img_path,"--------", np.array(img).shape, origin_w, origin_h)
                        imgs.append(transform(img))  # list[img]

                    imgs = torch.stack(imgs, dim=0).to(args.device)  # [clip_len, 3, h, w]
                    img_h, img_w = imgs.shape[-2:]
                    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                    target = {"size": size}
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

                    pred_masks = F.interpolate(seg_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)# t, c, h , w
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

    
    # print(fps_frames/frame_time)
   
   





if __name__ == '__main__':
    parser = argparse.ArgumentParser('OnlineRefer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    args.log_path, args.val_vis = set_path(args)
    main(args)
