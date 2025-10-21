python3 infer_ytvos_MS.py \
--device cuda:0 \
--exps "" \
--resume < ckpt path > \
--num_frames 72 \
--output_dir < path to save predictions > \
--ytvos_path < path to ref-yt data > \
--name_prefix < run name >