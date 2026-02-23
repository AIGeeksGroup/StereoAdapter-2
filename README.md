# <img src="./assets/stereoadapter_logo.png" alt="logo" width="50"/> StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation

This is the official repository for the paper:
> **StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation**
>
> [Zeyu Ren](https://github.com/Zephyr0609), Xiang Li, [Yiran Wang](https://github.com/u7079256), [Zeyu Zhang](https://steve-zeyu-zhang.github.io/), and [Hao Tang](https://ha0tang.github.io/)
>
> ### [Paper](https://arxiv.org/abs/2602.16915) | [Website](https://aigeeksgroup.github.io/StereoAdapter-2/)

## ✏️ Citation

If you find our code or paper helpful, please consider starring ⭐ us and citing:

```bibtex
@misc{ren2026stereoadapter2globallystructureconsistentunderwater,
  title={StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation},
  author={Zeyu Ren and Xiang Li and Yiran Wang and Zeyu Zhang and Hao Tang},
  year={2026},
  eprint={2602.16915},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.16915},
}
```

## 🔧 Quick Start

### 1. Install & Requirements

```bash
conda env create -f environment.yaml
conda activate stereoadapter2
```

### 2. Compile CUDA Kernels

Follow the installation instructions from [Mamba](https://github.com/state-spaces/mamba) and [VMamba](https://github.com/MzeroMiko/VMamba) to compile the required CUDA kernels.

### 3. Download Pretrained Depth Anything 3 Model

Download the pretrained weights from [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) or directly from [HuggingFace](https://huggingface.co/depth-anything/DA3-BASE):

```bash
mkdir -p depthanything/checkpoints
cd depthanything/checkpoints
hf download depth-anything/DA3-BASE --local-dir DA3-BASE
cd ../..
```

### 4. Train

```bash
bash train_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list.sh
```

## 😘 Acknowledgement

- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [StereoAdapter](https://github.com/AIGeeksGroup/StereoAdapter)

## 📜 License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
