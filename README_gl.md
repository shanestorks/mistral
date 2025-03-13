Below is the list of steps I took to get it working on Great Lakes.

1. First, load the appropriate modules for this code base (relies on CUDA 11.3):

```
module load python3.11-anaconda/2024.02
module load cuda/11.3.0
```

2. Create `mistral` environment:

```
conda create -n mistral python=3.8.12 pytorch=1.11.0 torchdata cudatoolkit=11.3 -c pytorch
source activate mistral
pip install -r setup/pip-requirements.txt
```

*Note: I ran these commands on the Great Lakes login node terminal - when trying to run them in an interactive session, I wound up with a corrupted installation of `torch` and the wrong Python version.*

3. 2 fixes to the environment to ensure compatibility among dependencies:

```
pip install --upgrade mkl==2024.0.0
pip install -U datasets
```

4. Configure `conf/mistral-micro.yaml` by setting `artifacts.cache_dir` and `artifacts.run_dir` to your desired paths.

5. Run single-node single-GPU training as follows:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/mistral-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --run_id tutorial-gpt2-micro
```

*Note: in addition to the above environment fixes, I had to change "wikitext" in `conf/datasets/wikitext2.yaml` to "Salesforce/wikitext".*