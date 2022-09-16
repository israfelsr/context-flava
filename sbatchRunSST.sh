export PYTHONPATH=$(pwd)
export PYTHONPATH=../context-flava/multimodal/examples/:$PYTHONPATH
python scripts/run_sst.py\
    config=../context-flava/multimodal/examples/flava/configs/finetuning/qnli.yaml