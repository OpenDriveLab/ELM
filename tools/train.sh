CONFIG=lavis/projects/blip2/train/advqa_t5_elm.yaml

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=10041 \
    train.py --cfg-path $CONFIG