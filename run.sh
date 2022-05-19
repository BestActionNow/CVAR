#!/bin/bash

python main.py --dataset_name movielens1M --model_name fm  --warmup_model base 
python main.py --dataset_name movielens1M --model_name fm  --warmup_model cvar --cvar_epochs 2 --cvar_iters 10
python main.py --dataset_name movielens1M --model_name dcn  --warmup_model base 
python main.py --dataset_name movielens1M --model_name dcn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 10
python main.py --dataset_name movielens1M --model_name ipnn  --warmup_model base 
python main.py --dataset_name movielens1M --model_name ipnn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 10
python main.py --dataset_name movielens1M --model_name opnn  --warmup_model base 
python main.py --dataset_name movielens1M --model_name opnn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 10

python main.py --dataset_name taobaoAD --model_name fm  --warmup_model base 
python main.py --dataset_name taobaoAD --model_name fm  --warmup_model cvar --cvar_epochs 2 --cvar_iters 1
python main.py --dataset_name taobaoAD --model_name dcn  --warmup_model base 
python main.py --dataset_name taobaoAD --model_name dcn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 1
python main.py --dataset_name taobaoAD --model_name ipnn  --warmup_model base 
python main.py --dataset_name taobaoAD --model_name ipnn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 1
python main.py --dataset_name taobaoAD --model_name opnn  --warmup_model base 
python main.py --dataset_name taobaoAD --model_name opnn  --warmup_model cvar --cvar_epochs 2 --cvar_iters 1


