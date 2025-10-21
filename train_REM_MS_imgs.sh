python3 train_REM_MS_imgs.py \
--dataset_file joint_im \
--device cuda:0 \
--num_frames 1 \
--clip_val 1e-1 \
--lr 1e-5 \
--output_dir < path to save training checkpoints and visuals > \
--coco_path < pat to refcoco data > \
--name_prefix < run name >