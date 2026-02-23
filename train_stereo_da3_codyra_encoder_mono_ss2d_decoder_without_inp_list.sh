#!/bin/bash
#SBATCH --job-name=stereoadapter2
#SBATCH --output=stereoadapter2_%j.log
#SBATCH --error=stereoadapter2_%j.log
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=20G

export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stereoadapter2

# CUDA environment
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR

echo "=== GPU Info ==="
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

# Compile sampler CUDA kernel
echo "=== Compiling sampler CUDA kernel ==="
if python -c "import corr_sampler" 2>/dev/null; then
    echo "corr_sampler already installed, skipping"
else
    cd $SLURM_SUBMIT_DIR/sampler && python setup.py install && cd $SLURM_SUBMIT_DIR
fi

# Compile selective_scan oflex kernel
echo "=== Compiling selective_scan oflex CUDA kernel ==="
cd $SLURM_SUBMIT_DIR/VMamba/kernels/selective_scan
pip uninstall -y selective_scan 2>/dev/null || true
rm -rf build dist *.egg-info 2>/dev/null || true
python setup.py install
cd $SLURM_SUBMIT_DIR

# Verify kernel installation
python -c "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex installed')"
if [ $? -ne 0 ]; then
    echo "selective_scan_cuda_oflex installation failed"
    exit 1
fi

# Train
LOG_DIR="./train_log"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting training on $(hostname) at $(date)"

python train_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list.py \
  --name stereoadapter2 \
  --train_datasets tartan_air \
  --da3_pretrained depthanything/checkpoints/DA3-BASE \
  --image_size 480 640 \
  --batch_size 8 \
  --num_steps 100000 \
  --train_iters 22 \
  --valid_iters 32 \
  --validation_frequency 10000 \
  --spatial_scale -0.2 0.4 \
  --saturation_range 0 1.4 \
  --n_downsample 2 \
  --lr 0.0001 \
  --use_mde_init \
  --baseline 0.25 \
  --focal 320.0 \
  2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

echo "Training completed at $(date)"
