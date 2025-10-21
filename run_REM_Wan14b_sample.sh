python3 infer_sample_wan.py \
--device cuda:0 \
--exps "referring exp 1" "referring exp 2"\
--base_dir < Wan_base_path > \
--resume < ckpt_path > \
--frame_dir < path to input frames > \
--num_frames 73 \
--output_dir < path to save predicted masks > \
--name_prefix < sample_name >