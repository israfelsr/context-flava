export PYTHONPATH=$(pwd)
export PYTHONPATH=../context-flava/multimodal/examples/:$PYTHONPATH
python scripts/run_sst.py\
    config=./configs/finetune/unimodal_sst.yaml\
    model.pretrained_model_key=flava_full