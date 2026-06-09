# PKGD

Official implementation of **PKGD: Physics Knowledge-Guided Diffusion Model for Underwater Image Enhancement**.

PKGD enhances degraded underwater images by injecting underwater imaging priors into a conditional diffusion model. The framework contains three main components:

<img width="1240" height="774" alt="Framework" src="https://github.com/user-attachments/assets/6d5541a5-e288-4331-bad7-043fb31193a6" />

## Environment

The code is developed with Python and PyTorch. A CUDA-enabled GPU is recommended for training and inference.

Install PyTorch and torchvision first according to your CUDA version. For example:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you want to use Weights & Biases logging, install it separately:

```bash
pip install wandb
```

## Datasets

The paper evaluates PKGD on underwater image enhancement benchmarks including **LSUI**, **UIEB**, and **U60**.

For paired training and validation, organize each dataset directory as follows. The default configuration uses `l_resolution=16` and `r_resolution=256`, so the input directory must be named `sr_16_256` and the ground-truth directory must be named `hr_256`.

```text
DatasetName_train_16_256/
|-- hr_256/       # ground-truth/reference images
`-- sr_16_256/    # degraded/input underwater images

DatasetName_val_16_256/
|-- hr_256/
`-- sr_16_256/
```

All paired images should be aligned and have the same filenames or the same sorted order. The current configuration assumes 256 x 256 images.

## Configuration

Edit `config/underwater.json` before running.

Important fields:

- `path.resume_state`: checkpoint prefix for resuming or inference. For a checkpoint saved as `I1550000_E1449_gen.pth`, set this to the path without `_gen.pth`, for example `experiments_train/PKGD_xxx/checkpoint/I1550000_E1449`.
- `datasets.train.dataroot`: path to `DatasetName_train_16_256`.
- `datasets.val.dataroot`: path to `DatasetName_val_16_256`.
- `datasets.*.l_resolution`: low-resolution/input tag used in the `sr_16_256` directory name.
- `datasets.*.r_resolution`: output/reference resolution used in the `hr_256` directory name.
- `model.beta_schedule.*.n_timestep`: diffusion timestep number for training and validation.
- `path.stage`: output stage name used to create `experiments_train` or `experiments_val`.

## Training

Set `path.resume_state` to `null`, fill in the training and validation dataset paths, and run:

```bash
python train.py -c config/underwater.json -p train -gpu 0
```

Logs, TensorBoard files, validation results, and checkpoints will be saved under:

```text
experiments_train/PKGD_<timestamp>/
```

To resume training, set `path.resume_state` to the checkpoint prefix and run the same command.

## Inference

Set:

- `path.stage` to `val`
- `path.resume_state` to the checkpoint prefix
- `datasets.val.dataroot` to the validation/test dataset directory

Then run:

```bash
python infer.py -c config/underwater.json -p val -gpu 0
```

The generated images are saved under:

```text
experiments_val/PKGD_<timestamp>/results/
```

Each sample may include:

- `*_sr.png`: final enhanced result
- `*_sr_process.png`: diffusion sampling process grid
- `*_inf.png`: input degraded image
- `*_hr.png`: reference image, when available

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{shi2026pkgd,
  title={PKGD: Physics Knowledge-Guided Diffusion Model for Underwater Image Enhancement},
  author={Shi, Lin and Yang, Yunhui and Liu, Yi and Wang, Huibing and Fu, Xianping and Jiang, Guangqi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## License

This project is released under the license in `LICENSE`.
