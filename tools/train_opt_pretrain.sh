CODE_HOME=/cpfs01/user/huanglinyan/projects/EmbodiedScene
CONFIG=/cpfs01/user/huanglinyan/projects/EmbodiedScene/lavis/projects/blip2/train/advqa_t5_opt.yaml

# cp -r /cpfs01/user/huanglinyan/cache/* ~/.cache

# cd到bevformer根目录
cd $CODE_HOME

python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=10043 \
    train.py --cfg-path $CONFIG


# python -m torch.distributed.run \
#     --nproc_per_node=8 \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=10042 \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     train.py --cfg-path $CONFIG 