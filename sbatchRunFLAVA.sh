export PYTHONPATH=$(pwd)
export PYTHONPATH=../context-flava/multimodal/examples/:$PYTHONPATH
export PYTHONPATH=../context-flava/multimodal/:$PYTHONPATH
python scripts/run_flava.py\
    config=./configs/finetuning/mm_sst2_test.yaml\
    model.pretrained_model_key=flava_full