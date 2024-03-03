CODE_HOME=/cpfs01/user/huanglinyan/projects/EmbodiedScene
CONFIG=/cpfs01/user/huanglinyan/projects/EmbodiedScene/lavis/projects/blip2/ablation/advqa_t5_llama.yaml

# cp -r /cpfs01/user/huanglinyan/cache/* /home/huanglinyan/.cache
# cp -r /cpfs01/user/huanglinyan/cache/* ~/.cache

# cd到bevformer根目录
cd $CODE_HOME

python -m torch.distributed.run \
    --nproc_per_node=8 \
    train.py --cfg-path $CONFIG