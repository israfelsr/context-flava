export PYTHONPATH=$(pwd)
python scripts/precontext/sst_to_image.py\
    --hf_repository=israfelsr/multimodal_sst2\
    --split=train\
    --num_chunks=5\
    --auth_token=