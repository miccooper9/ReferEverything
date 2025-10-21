python3 infer_ytvos_wan.py \
--device cuda:0 \
--exps "" \
--base_dir < path to base model > \
--resume < ckpt path > \
--num_frames 73 \
--output_dir < path to save predictions > \
--ytvos_path < path to ref-yt data > \
--name_prefix < run name >