CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=6 \
    train_REM_wan.py \
    --token_length 77 \
    --num_frames 17 \
    --lr 1e-5 \
    --name_prefix < run name > \
    --output_dir < path to save training checkpoints and visuals > \
    --T5_enc_path < path to T5 text embeddings > \
    --ytvos_path < path to ref-yt data > \
    --coco_path < pat to refcoco data > \
    --clip_val 1e-1 