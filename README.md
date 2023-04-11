# minimum testing for distributed torch training
- this repo provide a simple testing code for MNIST training 
torchrun \
    --standalone     \
    --nnodes=1      \
    --nproc_per_node=2 \
main.py \
    --batch_size 256 \
2> train_error.log 1>train_logs.log