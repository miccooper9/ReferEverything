CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_REM_MS_vid.py \
    --dataset_file ytvos_clip \
    --token_length 77 \
    --num_frames 8 \
    --lr 1e-6 \
    --name_prefix < run name > \
    --output_dir < path to save training checkpoints and visuals > \
    --ytvos_path < path to ref-yt data > \
    --coco_path < pat to refcoco data > \
    --clip_val 1e-1 