python3 infer_sample_MS.py \
--device cuda:0 \
--exps  "referring exp 1" "referring exp 2" \
--resume < ckpt_path > \
--frame_dir < path to input frames > \
--num_frames 72 \
--output_dir < path to save predicted masks > \
--name_prefix < sample_name >
