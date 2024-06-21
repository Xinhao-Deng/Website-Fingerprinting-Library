# Website-Fingerprinting-Library (WFlib)

<p align="center">
<img src=".\figures\wflib.jpg" height = "180" alt="" align=center />
<br><br>
</p>


WFlib is a Pytorch-based open-source library for website fingerprinting attacks, intended for research purposes only.

We provide a neat code base to evaluate 11 advanced DL-based WF attacks on multiple datasets. This library is derived from our ACM CCS 2024 paper. If you find this repo useful, please cite our paper.

```bibtex
@inproceedings{deng2024wflib,
  title={Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis},
  author={Deng, Xinhao and Li, Qi and Xu, Ke},
  booktitle={Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security},
  year={2024}
}
```

## WFlib Overview

The code library includes 11 DL-based website fingerprinting attacks.

| Attacks | Conference  | Paper | Code |
|----------|----------|----------|----------|
| AWF | NDSS 2018 | [Automated Website Fingerprinting through Deep Learning](https://arxiv.org/pdf/1708.06376) | [DLWF](https://github.com/DistriNet/DLWF) |
| DF | CCS 2018 | [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768) | [df](https://github.com/deep-fingerprinting/df) |
| Tik-Tok | PETS 2019 | [Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks](https://petsymposium.org/popets/2020/popets-2020-0043.pdf) | [Tik_Tok](https://github.com/msrocean/Tik_Tok) |
| Var-CNN | PETS 2019 | [Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning](https://arxiv.org/pdf/1802.10215) | [Var-CNN](https://github.com/sanjit-bhat/Var-CNN) |
| TF | CCS 2019 | [Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning](https://dl.acm.org/doi/pdf/10.1145/3319535.3354217) | [tf](https://github.com/triplet-fingerprinting/tf) |
| BAPM | ACSAC 2021 | [BAPM: Block Attention Profiling Model for Multi-tab Website Fingerprinting Attacks on Tor](https://dl.acm.org/doi/pdf/10.1145/3485832.3485891) | None |
| ARES | S&P 2023 | [Robust Multi-tab Website Fingerprinting Attacks in the Wild](http://www.thucsnet.com/wp-content/papers/xinhao_sp2023.pdf) | [Multitab-WF-Datasets](https://github.com/Xinhao-Deng/Multitab-WF-Datasets) |
| RF | Security 2023 | [Subverting Website Fingerprinting Defenses with Robust Traffic Representation](https://www.usenix.org/system/files/sec23fall-prepub-621_shen-meng.pdf) | [RF](https://github.com/robust-fingerprinting/RF) |
| NetCLR | CCS 2023 | [Realistic Website Fingerprinting By Augmenting Network Trace](https://arxiv.org/pdf/2309.10147) | [Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces](https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces) |
| TMWF | CCS 2023 | [Transformer-based Model for Multi-tab Website Fingerprinting Attack](https://dl.acm.org/doi/abs/10.1145/3576915.3623107) | [TMWF](https://github.com/jzx-bupt/TMWF) |
| Holmes | CCS 2024 | [Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library)|


We implemented all attacks using the same framework (Pytorch) and a consistent coding style, enabling researchers to evaluate and compare existing attacks easily.

## Usage

### Install 

```sh
pip install --user .
```

If you modify the core code of WFlib (e.g., code in `./models` or `./tools`), WFlib needs to be reinstalled.

### Datasets

- Download datasets ([link](https://drive.google.com/file/d/1yJJ7Qyba-9HF7MBgpFrkvfY4_3ZwTvyx/view?usp=sharing)) and place it in the folder `./datasets`

- Divide the dataset into training, validation, and test sets. 
For example, you can execute the following command.

```sh
python exp/dataset_process/dataset_split.py --dataset DF18
```

### Training \& Evaluation

We provide all experiment scripts for WF attacks on multiple datasets in the folder `./scripts/`. For example, you can reproduce the AWF attack on the DF18 dataset by executing the following command.

```sh
bash scripts/DF18/AWF.sh
```

The `./scripts/DF18/AWF.sh` file contains the commands for model training and evaluation.

```sh
python -u exp/train.py \
  --dataset DF18 \
  --model AWF \
  --gpu 0 \
  --feature DIR \
  --seq_len 3000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 8e-4 \
  --optimizer RMSprop \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset DF18 \
  --model AWF \
  --gpu 0 \
  --feature DIR \
  --seq_len 3000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_name max_f1
```

The meanings of all parameters can be found in the `exp/train.py` and `exp/test.py` files. WFlib supports modifying parameters to easily implement different attacks. Additionally, you can use WFlib to implement combinations of different attacks or perform ablation analysis.

## Contact
If you have any questions or suggestions, feel free to contact:

- [Xinhao Deng](https://xinhao-deng.github.io/) (dengxh23@mails.tsinghua.edu.cn)

## Acknowledgements

We would like to thank all the authors of the referenced papers. Special thanks to **Yixiang Zhang** and **Jie Yan** from Tsinghua University for their participation in the code review.