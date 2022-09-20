export PYTHONPATH=$(pwd)
python scripts/precontext/sst_to_image.py\
    --split=testing\
    --num_chunks=5\
    --auth_token=