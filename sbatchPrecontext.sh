export PYTHONPATH=$(pwd)
export PYTHONPATH=../context-flava/BLIP/:$PYTHONPATH
python scripts/precontext/imagenet_to_text.py\
    config=./configs/precontext/sst2_to_image.yaml