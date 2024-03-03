CONFIG=/cpfs01/user/huanglinyan/projects/EmbodiedScene/lavis/projects/blip2/train/advqa_t5_emdmulti.yaml

python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=10041 \
    train.py --cfg-path $CONFIG