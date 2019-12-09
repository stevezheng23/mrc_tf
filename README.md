# Machine Reading Comprehension
Machine reading comprehension (MRC), a task which asks machine to read a given context then answer questions based on its understanding, is considered one of the key problems in artificial intelligence and has significant interest from both academic and industry. Over the past few years, great progress has been made in this field, thanks to various end-to-end trained neural models and high quality datasets with large amount of examples proposed.
<p align="center"><img src="/docs/squad.example.png" width=800></p>
<p align="center"><i>Figure 1: MRC example from SQuAD 2.0 dev set</i></p>

## Setting
* Python 3.6.7
* Tensorflow 1.13.1
* NumPy 1.13.3
* SentencePiece 0.1.82

## DataSet
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
* [CoQA](https://stanfordnlp.github.io/coqa/) a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation. CoQA is pronounced as coca
* [QuAC](https://quac.ai/) is a dataset for modeling, understanding, and participating in information seeking dialog. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context.

## Usage
* Run SQuAD experiment
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad.py \
    --spiece_model_file=model/cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=model/cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=model/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --task_name=v2.0 \
    --random_seed=100 \
    --predict_tag=xxxxx \
    --data_dir=data/squad/v2.0 \
    --output_dir=output/squad/v2.0/data \
    --model_dir=output/squad/v2.0/checkpoint \
    --export_dir=output/squad/v2.0/export \
    --max_seq_length=512 \
    --train_batch_size=12 \
    --predict_batch_size=12 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=true \
    --do_predict=true \
    --do_export=true \
    --overwrite_data=false
```
* Run CoQA experiment
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_coqa.py \
    --spiece_model_file=model/cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=model/cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=model/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --task_name=v1.0 \
    --random_seed=100 \
    --predict_tag=xxxxx \
    --data_dir=data/coqa/v1.0 \
    --output_dir=output/coqa/v1.0/data \
    --model_dir=output/coqa/v1.0/checkpoint \
    --export_dir=output/coqa/v1.0/export \
    --max_seq_length=512 \
    --train_batch_size=12 \
    --predict_batch_size=12 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=true \
    --do_predict=true \
    --do_export=true \
    --overwrite_data=false
```
* Run QuAC experiment
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_quac.py \
    --spiece_model_file=model/cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=model/cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=model/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --task_name=v1.0 \
    --random_seed=100 \
    --predict_tag=xxxxx \
    --data_dir=data/quac/v0.2 \
    --output_dir=output/quac/v0.2/data \
    --model_dir=output/quac/v0.2/checkpoint \
    --export_dir=output/quac/v0.2/export \
    --max_seq_length=512 \
    --train_batch_size=12 \
    --predict_batch_size=12 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=true \
    --do_predict=true \
    --do_export=true \
    --overwrite_data=false
```

## Experiment
### SQuAD v1.1
<p align="center"><img src="/docs/xlnet.squad.v1.png" width=500></p>
<p align="center"><i>Figure 2: Illustrations of fine-tuning XLNet on SQuAD v1.1 task</i></p>

|       Model       | Train Data | # Epoch | # Train Steps | Batch Size | Max Length | Learning Rate |    EM    |    F1    |
|:-----------------:|:----------:|:-------:|:-------------:|:----------:|:----------:|:-------------:|:--------:|:--------:|
|     XLNet-base    |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   85.90  |   92.17  |
|     XLNet-large   |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   88.61  |   94.28  |

<p><i>Table 1: The dev set performance of XLNet model finetuned on SQuAD v1.1 task</i></p>

### SQuAD v2.0
<p align="center"><img src="/docs/xlnet.squad.v2.png" width=500></p>
<p align="center"><i>Figure 3: Illustrations of fine-tuning XLNet on SQuAD v2.0 task</i></p>

|       Model       | Train Data | # Epoch | # Train Steps | Batch Size | Max Length | Learning Rate |    EM    |    F1    |
|:-----------------:|:----------:|:-------:|:-------------:|:----------:|:----------:|:-------------:|:--------:|:--------:|
|     XLNet-base    |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   80.23  |   82.90  |
|     XLNet-large   |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   85.72  |   88.36  |

<p><i>Table 2: The dev set performance of XLNet model finetuned on SQuAD v2.0 task</i></p>

## Reference
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. [SQuAD: 100,000+ questions for machine comprehension of text](https://arxiv.org/abs/1606.05250) [2016]
* Pranav Rajpurkar, Robin Jia, and Percy Liang. [Know what you don’t know: unanswerable questions for SQuAD](https://arxiv.org/abs/1806.03822) [2018]
* Siva Reddy, Danqi Chen, Christopher D. Manning. [CoQA: A Conversational Question Answering Challenge](https://arxiv.org/abs/1808.07042) [2018]
* Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, Luke Zettlemoyer. [QuAC : Question Answering in Context](https://arxiv.org/abs/1808.07036) [2018]
* Danqi Chen. [Neural reading comprehension and beyond](https://cs.stanford.edu/~danqi/papers/thesis.pdf) [2018]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le and Ruslan Salakhutdinov. [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860) [2019]
