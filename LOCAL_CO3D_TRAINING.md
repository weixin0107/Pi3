# CO3D 本地小规模训练指南

本文记录在单张 NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存）上，使用已有的 `.venv` 和本地 CO3D 数据进行 π³ 小规模从零训练的环境适配、启动流程及验证结果。

## 1. 当前环境

- 项目目录：`/home/voyah/Desktop/Pi3`
- Python 环境：项目内已有的 `.venv`
- Python：3.10
- PyTorch：2.5.1+cu124
- CUDA：12.4
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8 GB
- 混合精度：BF16
- CO3D 数据目录：`/home/voyah/Downloads/co3d`

本流程不会创建或重建虚拟环境，所有 Python 命令和依赖安装均明确指向现有 `.venv`。

## 2. 验证现有环境

在项目根目录执行：

```bash
.venv/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

预期结果包括：

- Python 路径为项目中的 `.venv/bin/python`
- `torch.cuda.is_available()` 为 `True`
- GPU 为 RTX 4060 Laptop GPU
- BF16 支持为 `True`

还可以检查驱动和空闲显存：

```bash
nvidia-smi
```

## 3. 在已有 `.venv` 中补充依赖

训练器通过 `transformers.trainer_pt_utils.get_model_param_count` 统计模型参数，因此需要安装 `transformers`。

使用 `uv` 将依赖安装到现有环境：

```bash
uv pip install --python .venv/bin/python transformers
```

本次安装版本为 `transformers==5.17.0`。项目的 `requirements.txt` 已同步加入 `transformers`。

如果以后需要补齐依赖，同样可以直接安装到现有环境：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## 4. CO3D 数据准备

本次使用的数据根目录为：

```text
/home/voyah/Downloads/co3d
```

当前包含以下类别：

- `apple`
- `baseballglove`
- `microwave`
- `parkingmeter`
- `tv`

数据根目录中包含各类别的训练和测试标注，例如：

```text
apple_train.jgz
apple_test.jgz
baseballglove_train.jgz
baseballglove_test.jgz
...
```

类别目录中应包含图像、深度图和掩码等标注所引用的文件。当前数据约 168 GB。

项目已有以下缓存：

```text
data/dataset_cache/co3dv2_train_cache.npy
data/dataset_cache/co3dv2_test_cache.npy
```

已验证缓存包含：

- 训练集：1005 个视频序列
- 测试集：99 个视频序列
- 缓存中的类别与本地标注文件一致

如果更换数据类别或标注文件，应删除上述缓存，让程序按新数据重新生成，避免缓存指向不存在的类别。

## 5. 本地数据配置

本地数据配置位于：

```text
configs/data/co3d_local.yaml
```

主要设置如下：

- `data_root: /home/voyah/Downloads/co3d`
- 训练和测试均使用 `CO3DV2Dataset`
- 每个样本最多读取 4 帧
- 训练集启用裁剪、图像增强和随机背景掩码
- 测试分辨率为 224×224

## 6. 8 GB 显存模型适配

### 6.1 原始模型无法直接从零训练

原始 large 模型约有 8.92 亿个可训练参数。即使使用 BF16，AdamW 在第一次更新时仍需创建一阶和二阶动量状态，实测在 RTX 4060 8 GB 上发生 CUDA OOM：

```text
Total Trainable Params: 892366936
torch.OutOfMemoryError: CUDA out of memory
```

仅降低帧数或图像分辨率无法解决优化器状态占用，因此本地流程改为训练缩小后的完整模型，而不是冻结随机初始化的编码器。

### 6.2 小型编码器和解码器

`pi3/models/pi3_training.py` 已增加 `encoder_size` 参数，支持：

- `small`：DINOv2 ViT-S/14，特征维度 384
- `base`：DINOv2 ViT-B/14，特征维度 768
- `large`：DINOv2 ViT-L/14，特征维度 1024

编码器与主解码器的尺寸必须一致，以确保 token 特征维度能够拼接。默认配置仍保持 `large`，不会改变原始训练流程。

VGGT 权重只与 large 模型兼容；选择 small 或 base 时必须关闭 `load_vggt`。

### 6.3 本地训练模型设置

本地训练配置位于：

```text
configs/train/train_pi3_lowres_local.yaml
```

模型设置为：

```yaml
model:
    encoder_size: small
    decoder_size: small
    load_vggt: false
    freeze_encoder: false
    ckpt: null
```

这些设置表示：

- 使用 small 编码器和 small 主解码器
- 不依赖 `ckpts/VGGT-1B/model.safetensors`
- 编码器不冻结
- 不加载已有 π³ 检查点
- 所有约 1.96 亿参数均从零训练

实测峰值显存约 3693 MiB，适合当前 8 GB GPU。

## 7. 小规模训练参数

`configs/train/train_pi3_lowres_local.yaml` 中的主要参数：

```yaml
train:
    image_num_range: [2, 4]
    max_img_per_gpu: 4
    resolution:
        - [224, 224]
    num_workers: 2
    num_epoch: 1
    iters_per_epoch: 10
    print_freq: 1

test:
    num_workers: 2
    iters_per_test: 2
    print_freq: 1
```

这是用于验证环境、数据、前向传播、反向传播和检查点保存是否正常的短流程，不代表正式收敛训练所需的训练量。

注意：当前验证循环没有根据 `test.iters_per_test` 提前退出，因此本次实际遍历了完整的本地测试加载器，共输出 100 个验证批次。该行为不影响训练正确性，但会使验证次数多于配置值。

## 8. 启动训练

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HYDRA_FULL_ERROR=1 \
.venv/bin/accelerate launch \
  --config_file configs/accelerate/local_single_gpu.yaml \
  scripts/train_pi3.py \
  train=train_pi3_lowres_local \
  data=co3d_local \
  name=pi3_co3d_scratch_small
```

参数说明：

- `CUDA_VISIBLE_DEVICES=0`：只使用第 1 张 GPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：减少显存碎片
- `HYDRA_FULL_ERROR=1`：发生错误时显示完整调用栈
- `local_single_gpu.yaml`：单机、单进程、BF16 配置
- `train=train_pi3_lowres_local`：使用本地小规模训练配置
- `data=co3d_local`：使用本地 CO3D 数据配置
- `name=pi3_co3d_scratch_small`：决定输出目录名称

命令直接调用 `.venv/bin/accelerate`，不依赖 shell 中当前激活的是哪个 Python 环境。

## 9. 本次训练结果

本次短训练已成功完成，进程退出码为 0：

- 可训练参数：196,493,016
- 训练轮数：1
- 训练步数：10
- 训练精度：BF16
- 峰值显存：约 3693 MiB
- 最终验证损失：约 0.7095
- 总耗时：约 36 秒

输出目录：

```text
outputs/pi3_co3d_scratch_small/
```

主要文件：

```text
outputs/pi3_co3d_scratch_small/log.log
outputs/pi3_co3d_scratch_small/ckpts/log.txt
outputs/pi3_co3d_scratch_small/ckpts/best_model/pytorch_model.bin
outputs/pi3_co3d_scratch_small/ckpts/best_model/optimizer.bin
outputs/pi3_co3d_scratch_small/ckpts/best_model/scheduler.bin
outputs/pi3_co3d_scratch_small/ckpts/checkpoint_0/pytorch_model.bin
outputs/pi3_co3d_scratch_small/ckpts/checkpoint_0/optimizer.bin
outputs/pi3_co3d_scratch_small/ckpts/checkpoint_0/scheduler.bin
```

`best_model` 是按验证损失保存的最佳状态，`checkpoint_0` 是第 1 个 epoch 结束时保存的完整训练状态。

## 10. 扩大训练规模

确认短流程稳定后，可以通过命令行覆盖训练量，而不修改配置文件。例如训练 10 个 epoch，每个 epoch 200 步：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/accelerate launch \
  --config_file configs/accelerate/local_single_gpu.yaml \
  scripts/train_pi3.py \
  train=train_pi3_lowres_local \
  data=co3d_local \
  name=pi3_co3d_scratch_small_long \
  train.num_epoch=10 \
  train.iters_per_epoch=200 \
  train.print_freq=10
```

建议使用新的 `name`，避免与短测试输出混合。训练规模增加前还应确认磁盘空间，因为每个完整检查点会同时保存模型、优化器和调度器状态。

## 11. 常见问题

### 缺少 `transformers`

错误：

```text
ModuleNotFoundError: No module named 'transformers'
```

解决：

```bash
uv pip install --python .venv/bin/python transformers
```

### 缺少 VGGT 权重

错误通常指向：

```text
ckpts/VGGT-1B/model.safetensors
```

本地从零训练配置已经设置 `load_vggt: false`，不应读取该文件。确认使用了 `train=train_pi3_lowres_local`。

### CUDA 显存不足

确认日志中的模型配置为：

```text
encoder_size: small
decoder_size: small
load_vggt: false
freeze_encoder: false
```

同时保留：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

如果仍然显存不足，可以将 `train.image_num_range` 和 `train.max_img_per_gpu` 都降到 2，但不要改成冻结随机初始化编码器来掩盖显存问题。

### RoPE2D CUDA 扩展警告

当前会看到：

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

程序会回退到 PyTorch 实现，本次训练已验证可以正常完成；该警告影响性能，不影响短流程功能验证。
