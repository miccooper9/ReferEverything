python3 infer_burst_wan.py \
--device cuda:0 \
--exps "" \
--base_dir < path to base model > \
--resume < ckpt path >  \
--num_frames 65 \
--output_dir < path to save predictions > \
--TAO_path < path to TAO/Burst data > \
--name_prefix < run name >