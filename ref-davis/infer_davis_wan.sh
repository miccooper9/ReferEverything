python3 infer_davis_wan.py \
--device cuda:0 \
--exps "" \
--base_dir < path to base model > \
--resume < ckpt path > \
--num_frames 73 \
--output_dir < path to save predictions > \
--name_prefix < run name >