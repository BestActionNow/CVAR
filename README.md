# CVAR

================================================

by XuZhao (<xuzzzhao@tencent.com>)

Overview
--------

PyTorch implementation of our paper "Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder" accepted by SIGIR 2022.


Dependencies
------------

Install Pytorch 1.10.0, using pip or conda, should resolve all dependencies.

Tested with Python 3.8.5, but should work with 3.x as well.

Tested with sklearn 0.0.

Tested on CPU or GPU.

Dataset
-------

You can download the datasets we introduced in our paper from following links:
* [Movielens1M](http://files.grouplens.org/datasets/movielens/)
* [TaobaoAD](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)

Raw data need to be preprocessed before using. The data preprocessing scripts are given in `datahub/movielens1M/movielens1M_preprocess.ipynb` and `datahub/taobaoAd/taobaoAD_preprocess.ipynb` for movielens1M and taobaoAD respectively.

How to Use
----------
`model/*`: Implementation of various backbone models.

`model/warm.py`: Implementation of three warm-up models. 

`main.py`: Start Point of experiment.

You can conduct experiments as following command:
<br>
<br>
`python main.py --dataset_name movielens1M  --model_name deepfm --warmup_model cvar  --cvar_iters 10`
<br>
<br>
`python main.py --dataset_name taobaoAD  --model_name deepfm  --warmup_model cvar  --cvar_iters 1`
<br>
<br>
Notice that the hyperparameter *--cvar_iters* is set 10 for movielens1M dataset while 1 for taobaoAD dataset. 

Moreover, the command to get every data point in our paper is given in  `./run.sh`, including some hyperparameters and random seed setting. 

The program will print the AUC, F1 in cold-start stage and three warm-up stages. Part of settable parameters are listed as follows:

Parameter | Options | Usage
--------- | ------- | -----
--dataset_name |  | Specify the dataset for evaluation
--dataset_path | | Specify the dataset path for evaluation
--model_name | [fm, deepfm, wd, dcn, ipnn, opnn] | Specify the backbone for recommendation 
--warmup_model |[base, mwuf, metaE, cvar_init, cvar] | Specify the warm-up method
--is_dropoutnet | [True, False] | Specify whether to use dropoutNet for backbone pretraining
--device | [cpu, cuda:0] | Specify the device (CPU or GPU) to run the program
--runs | | Specify the number of executions to compute average metrics

Some other settable parameters could be found in the `./main.py` file.


Citation
--------


If you want to refer to our work, please cite our paper as:
```
@article{
  title={Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder},
  author={Xu Zhao, Yi Ren, Ying Du, Shenzheng Zhang, Nian Wang},
  booktitle={SIGIR},
  year={2022},
}
```
