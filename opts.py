import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=3e-5, type=float)
    
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--amsgrad', action='store_true', help='if true, set amsgrad to True in an Adam or AdamW optimizer.')




    
    
    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--T5_enc_path', type=str, default='../txtT5') 
    parser.add_argument('--coco_path', type=str, default='../COCO-rvos')
    parser.add_argument('--ytvos_path', type=str, default='../rvos')
    parser.add_argument('--davis_path', type=str, default='../ref-davis')
    parser.add_argument('--davis_anno_path', type=str, default='../DAVIS')
    parser.add_argument('--TAO_path', type=str, default='../TAO')
    parser.add_argument('--vspw_path', type=str, default='../VSPW_480p/')
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--mix_frames', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--cogdres', action='store_true')
    parser.add_argument('--wanres', action='store_true')
    parser.add_argument('--clip_val', default=5.0, type=float)
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")

    parser.add_argument('--base_dir', default='../Wan2.1-T2V-14B-Diffusers',help='path where the base vgen model is downloaded')
    parser.add_argument('--frame_dir', default='output',help='path tp read input frames from')
    parser.add_argument('--output_dir', default='output',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--token_length', default=77, type=int)
    parser.add_argument('--t', default=20, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=3, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--name_prefix', default='debugbasic', type=str)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--overlay_mask', default=True, type=bool)

    parser.add_argument('--exps', type=str, nargs='+', help='List of referring text expressions')

    parser.add_argument('--num_frames', default=73, type=int, help="Number of clip frames for training")


    parser.add_argument('--sampler_interval', default=3, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--num_clips', default=1, type=int,
                        help="Number of clips for training, default=1 for online mode")

    return parser


