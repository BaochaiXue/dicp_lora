# Distillation for In-Context Planning (DICP)

This repository provides the official implementation of our ICLR 2025 paper, [Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning](https://openreview.net/forum?id=BfUugGfBE5&noteId=BfUugGfBE5).

See [implementation_overview.md](doc/implementation_overview.md) for details on how the code maps observations, actions and rewards to transformer tokens.

## Requirements

To set up the required environment, run:
```bash
conda env create -f environment.yml
```

### Meta-World Installation

The following command installs Meta-World, adapted from [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld):
```bash
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
git checkout 83ac03c
pip install .
cd .. && rm -rf Metaworld
```

### TinyLlama Dependencies

To install the required TinyLlama dependencies, run the following commands (adapted from [TinyLlama’s PRETRAIN.md](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)):
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout 320fb59
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../../.. && rm -rf flash-attention
```

### LoRA Fine-Tuning

For low-rank adaptation we rely on [PEFT](https://github.com/huggingface/peft). Install it with

```bash
pip install peft==0.15.2
```

Enable LoRA by setting `use_lora: true` in the chosen model YAML file and adjust `lora_r`, `lora_alpha` and `lora_dropout` as needed. During training the script will report the number of trainable parameters.

## Usage

The following commands demonstrate the basic usage of the code in GridWorld environments.

### Data Collection

To collect training data, run:
```bash
python collect_data.py -ac [algorithm config]  -ec [environment config] -t [trajectory directory]
```

### Training

To train the model, run:
```bash
python train.py -ac [algorithm config]  -ec [environment config] -mc [model config] -t [trajectory directory] -l [log directory]
```

### Evaluation

To evaluate a trained model, run:
```bash
python evaluate.py -c [checkpoint directory] -k [beam size]
```


## Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{son2025distilling,
  author    = {Jaehyeon Son and Soochan Lee and Gunhee Kim},
  title     = {Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
}
```
