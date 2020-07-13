# Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs

This is an implemetation of the paper [Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs](https://arxiv.org/abs/2007.02387).

## Pretrain files

The codes rely on pre-trained BERT models. Please download `pretrain.tar` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1) and put it under the root. Then run `tar xvf pretrain.tar` to decompress it.

## Usage

To run the model on the FewRel dataset, we could use the following command:
```
python train_demo.py --trainN 5 --N 5 --K 1 --Q 1 --model regrab --encoder bert --hidden_size 768 --val_step 1000 --batch_size 8 --fp16 --seed 1
```

## Acknowledgement

Most of the codes are from the [FewRel repo](https://github.com/thunlp/FewRel), which provides a neat codebase for few-shot relation extraction.

## Citation

Please consider citing the following paper if you find our codes helpful. Thank you!
```
@inproceedings{qu2020few,
title={Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs},
author={Qu, Meng and Gao, Tianyu and Xhonneux, Louis-Pascal AC and Tang, Jian},
booktitle={International Conference on Machine Learning},
year={2020}
}
```
