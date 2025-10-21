python3 infer_vspw_stuff_wan.py \
--device cuda:0 \
--exps "" \
--base_dir < path to base model > \
--resume < ckpt path >  \
--num_frames 105 \
--output_dir < path to save predictions > \
--vspw_path < path tp vspw data > \
--name_prefix < run name >