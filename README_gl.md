Below is the list of steps I took to get it working on Great Lakes.

1. Clone this forked repo in your folder of choice:

```
cd path/to/where/you/want/to/clone
git clone https://github.com/shanestorks/mistral.git
cd mistral
```

2. Load the appropriate modules for this code base (relies on CUDA 11.3):

```
module load python3.11-anaconda/2024.02
module load cuda/11.3.0
```

3. If needed, delete the old bugged version of the `mistral` environment (type `y` when prompted):
```
conda env remove -n mistral --all
```

4. Create `mistral` environment (type `y` when prompted):

```
conda create -n mistral python=3.8.12 pytorch=1.11.0 torchdata cudatoolkit=11.3 -c pytorch
source activate mistral
pip install -r setup/pip-requirements.txt
```

*Note: I ran these commands on the Great Lakes login node terminal - when trying to run them in an interactive session, I wound up with a corrupted installation of `torch` and the wrong Python version.*

5. 2 fixes to the environment to ensure compatibility among dependencies:

```
pip install --upgrade mkl==2024.0.0
pip install -U datasets
```

6. Configure `conf/mistral-small.yaml` by setting `artifacts.cache_dir` and `artifacts.run_dir` to your desired paths.

7. Log into weights and biases, or run `export DISABLE_WANDB=true`.

8. Run single-node single-GPU training as follows:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/mistral-small.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 32 --run_id tutorial-gpt2-small
```

*Note: in addition to the above environment fixes, I had to change "wikitext" in `conf/datasets/wikitext2.yaml` to "Salesforce/wikitext".*

*Another note: I upped the batch size in the above command to maximize the GPU memory. However, we may need to tune the learning rate later to ensure model convergence.*

9. Run multi-node, multi-GPU training using the below Slurm script:

```
#!/bin/bash
# JOB HEADERS HERE
#SBATCH --partition=spgpu
#SBATCH --time=48:00:00
#SBATCH --job-name=train_gpt2
#SBATCH --account=sstorks0
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g
#SBATCH --mail-type=BEGIN,END,FAIL

export NNODES=4 # Configure # nodes here, and also update "--nodes" parameter above to match
export NPROC_PER_NODE=2 # Configure # processes per node here, and also update "--gpus-per-node" parameter above to match

cd /nfs/turbo/coe-sstorks/sstorks/mistral
module load python3.11-anaconda/2024.02
module load cuda/11.3.0
source activate mistral

RDZV_ID=$RANDOM
export HOST_NODE=$(srun --nodes=1 --ntasks=1 hostname)

nvidia-smi
lspci | grep -i nvidia


                       
srun --cpus-per-task 4 torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --rdzv_id=$RDZV_ID --rdzv_backend=c10d --rdzv_endpoint="$HOST_NODE:25703" train.py --config conf/mistral-small.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 32 --run_id distributed-gpt2-small
```

**After initial setup, do this each time to prepare the environment:**

```
module load python3.11-anaconda/2024.02
module load cuda/11.3.0
source activate mistral
cd path/to/mistral
```
