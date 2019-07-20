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

## Usage
* Run train
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
    --predict_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=true \
    --do_predict=false \
    --do_export=false \
    --overwrite_data=false
```
* Run predict
```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
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
    --train_batch_size=48 \
    --predict_batch_size=32 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=false \
    --do_predict=true \
    --do_export=false \
    --overwrite_data=false
```
* Run export
```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
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
    --train_batch_size=48 \
    --predict_batch_size=32 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=3e-5 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=false \
    --do_predict=false \
    --do_export=true \
    --overwrite_data=false
```

## Experiment
### SQuAD v2.0
<p align="center"><img src="/docs/squad.xlnet.png" width=500></p>
<p align="center"><i>Figure 2: Illustrations of fine-tuning XLNet on SQuAD v2.0 task</i></p>

|       Model       | # Epoch | # Train Steps | Batch Size |   Max Length  | Learning Rate |   EM   |   F1   |
|:-----------------:|:-------:|:-------------:|:----------:|:-------------:|:-------------:|:------:|:------:|
|     XLNet-large   |    ~3   |     8,000     |     48     |      512      |      3e-5     |   N/A  |   N/A  |

<p><i>Table 1: The dev set performance of XLNet model finetuned on SQuAD v2.0 task</i></p>

## Reference
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. [SQuAD: 100,000+ questions for machine comprehension of text](https://arxiv.org/abs/1606.05250) [2016]
* Pranav Rajpurkar, Robin Jia, and Percy Liang. [Know what you don’t know: unanswerable questions for SQuAD](https://arxiv.org/abs/1806.03822) [2018]
* Danqi Chen. [Neural reading comprehension and beyond](https://cs.stanford.edu/~danqi/papers/thesis.pdf) [2018]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le and Ruslan Salakhutdinov. [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860) [2019]