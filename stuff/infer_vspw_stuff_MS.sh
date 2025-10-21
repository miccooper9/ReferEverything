python3 infer_vspw_stuff_MS.py \
--device cuda:0 \
--exps "" \
--resume < ckpt path >  \
--num_frames 32 \
--output_dir < path to save predictions > \
--vspw_path < path tp vspw data > \
--name_prefix < run name >