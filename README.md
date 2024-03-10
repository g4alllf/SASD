# Improving Transferable Targeted Adversarial Attacks with Model Self-Enhancement (CVPR 2024)

**Pytorch implementation for the Sharpness-Aware Self-Distillation attack(SASD).**

## Code
### Code structure
```
.
├── imagenet
├── NIPS17
├── results
├── src
│   ├── eval_ensemble.py
│   ├── eval_single.py
│   ├── finetune.py
│   ├── GN
│   ├── rap_baseline.py
│   ├── rap_ensemble_baseline.py
│   ├── SASD.py
│   ├── torch_nets
│   └── utils
│       ├── args.py
│       ├── attack_methods.py
│       ├── image_loader.py
│       ├── image_process.py
│       ├── model_loader.py
│       ├── SAM.py
│       ├── scale_weight.py
│       └── utils.py
└── tf2torch_models
```

### Prepare datasets and model checkpoints
Datasets and model checkpoints are available in [NIPS17](https://www.kaggle.com/competitions/nips-2017-targeted-adversarial-attack/data), [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/), and [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model).

SASD model's checkpoints can be downloaded here: [link](https://drive.google.com/drive/folders/1CsNN53GYy9nFcJdSkS5Pcy_faisMDRRh?usp=drive_link).

### Install dependencies
```
pip install -r requirements.txt
```

### Run SASD
Please add the directory to PYTHONPATH before running SASD:
```
cd SASD
export PYTHONPATH="$PYTHONPATH:$PWD"
```

When the ILSVRC2012 and NIPS17 datasets are available, you can run SASD and check it's performance following the argparser suggestions in **src/utils/args.py**.

Here is an sample command for running Sharpness-Aware Self-Minimization(SASD):
```
python src/SASD.py \
    -m resnet50 \
    -b 10 \
    -e 2 \
    -t 1 \
    -w 20 \
    --sharpness_aware \
    -o 1 \
    -l 0.05
```

## Acknowledgements
This project make use of the following third-party projects:

1. **[SAM](https://github.com/davda54/sam)** We referenced the SAM optimizer settings from this code repository.
2. **[Targeted Transfer](https://github.com/ZhengyuZhao/Targeted-Transfer)** We referenced the implementation of generating targeted adversarial perturbations from this code repository.

We used the methods from the following repository when comparing the baseline.

1. **[GhostNet](https://github.com/LiYingwei/ghost-network)**
2. **[RAP](https://github.com/SCLBD/Transfer_attack_RAP)**
3. **[LGV](https://github.com/Framartin/lgv-geometric-transferability)**

We'd like to express our gratitude to the authors of these projects for their work, which greatly facilitated the development of this project.
