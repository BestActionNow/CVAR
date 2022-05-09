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
Tested with sklearn 0.0
Tested on CPU or GPU.

Dataset
-------

You can download the datasets we introduced in our paper from following links:
* [Movielens1M](http://files.grouplens.org/datasets/movielens/)
* [TaobaoAd](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)

Raw data need to be preprocessed before using. The data preprocessing scripts are given in `datahub/movielens1M/ml-1m_preprocess.ipynb` and `datahub/taobaoAd/taobao-ad_preprocess.ipynb` for movielens1M and taobaoAd respectively.

How to Use
----------
`model/*`: Implementation of various backbone models.

`model/warm.py`: Implementation of three warm-up models. 

`main.py`: Start Point of experiment.

You can conduct experiments as following command:
<br>
<br>
`python main.py --model_name deepfm --dataset_name movielens1M --dataset_path ./datahub/movielens1M/ml-1M.pkl --warmup_model cvar`
<br>
<br>
The program will print the AUC, F1 in cold-start stage and three warm-up stages. Part of settable parameters are listed as follows:

Parameter | Options | Usage
--------- | ------- | -----
--dataset_name |  | Specify the dataset for evaluation
--dataset_path | | Specify the dataset path for evaluation
--model_name | [afm, afn, dcn, deepfm, fm, pnn, wide&deep] | Specify the backbone for recommendation 
--warmup_model |[mwuf, metaE, cvar] | Specify the warm-up method
--is_dropoutnet | [True, False] | Sepficy whether to use dropoutNet for backbone pretraining

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
