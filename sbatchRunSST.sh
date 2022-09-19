export PYTHONPATH=$(pwd)
export PYTHONPATH=../context-flava/multimodal/examples/:$PYTHONPATH
python scripts/run_sst.py\
    config=./configs/finetuning/unimodal_sst2.yaml\
    model.pretrained_model_key=flava_full