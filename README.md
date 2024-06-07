# FedGT: Federated Node Classification with Scalable Graph Transformer

Preliminary version of code: https://arxiv.org/html/2401.15203v1.

<div align=center><img src="https://github.com/zaixizhang/FedGT/blob/main/fedgt_model.png" width="700"/></div>
The framework of FedGT. We use a case with three clients for illustration and omit the model details of Client 2 for simplicity. The node colors indicate the node labels.

## Requirement
- Python 3.9.16
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0
- METIS (for data generation), https://github.com/james77777778/metis_python

## Data Generation
Following command lines automatically generate the dataset.
```sh
$ cd data/generators
$ python disjoint.py
$ python overlapping.py
```

## Run 
Following command lines run the experiments for both FedAvg and our FED-PUB.
```sh
$ sh ./scripts/disjoint.sh [gpus] [num_workers]
$ sh ./scripts/overlapping.sh [gpus] [num_workers]
```

- `gpus`: specify gpus to use
- `num workers`: specify the number of workers on gpus (e.g. if your experiment uses 10 clients for every round then use less than or equal to 10 workers). The actual number of workers will be `num_workers` + 1 (one additional worker for a server).

Example
```sh
$ sh ./scripts/disjoint.sh 0,1 10
$ sh ./scripts/overlapping.sh 0,1 10
```

## Citation

If you found the paper and code useful to you, you can kindly cite our paper. Thanks! </br>

```BibTex
@article{zhang2024fedgt,
  title={FedGT: Federated Node Classification with Scalable Graph Transformer},
  author={Zhang, Zaixi and Hu, Qingyong and Yu, Yang and Gao, Weibo and Liu, Qi},
  journal={arXiv preprint arXiv:2401.15203},
  year={2024}
}
```

## Acknowledgement

This project draws in part from [FED-PUB](https://github.com/JinheonBaek/FED-PUB). Thanks for their great work and code!
