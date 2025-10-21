python3 infer_burst_MS.py \
--device cuda:0 \
--exps "" \
--resume < ckpt path > \
--num_frames 64 \
--output_dir < path to save predictions > \
--TAO_path < path to TAO/Burst data > \
--name_prefix < run name >