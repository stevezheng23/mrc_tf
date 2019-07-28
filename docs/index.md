## Description
Machine reading comprehension (MRC), a task which asks machine to read a given context then answer questions based on its understanding, is considered one of the key problems in artificial intelligence and has significant interest from both academic and industry. Over the past few years, great progress has been made in this field, thanks to various end-to-end trained neural models and high quality datasets with large amount of examples proposed.

![squad_example]({{ site.url }}/mrc_tf/squad.example.png){:width="800px"}

*Figure 1: MRC example from SQuAD 2.0 dev set*

## DataSet
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

## Experiment
### SQuAD v1.1

![xlnet_squad_v1]({{ site.url }}/mrc_tf/xlnet.squad.v1.png){:width="500px"}

*Figure 2: Illustrations of fine-tuning XLNet on SQuAD v1.1 task*

|       Model       | Train Data | # Epoch | # Train Steps | Batch Size | Max Length | Learning Rate |    EM    |    F1    |
|:-----------------:|:----------:|:-------:|:-------------:|:----------:|:----------:|:-------------:|:--------:|:--------:|
|     XLNet-base    |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   85.90  |   92.17  |
|     XLNet-large   |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   88.61  |   94.28  |

*Table 1: The dev set performance of XLNet model finetuned on SQuAD v1.1 task*

### SQuAD v2.0

![xlnet_squad_v2]({{ site.url }}/mrc_tf/xlnet.squad.v2.png){:width="500px"}

*Figure 3: Illustrations of fine-tuning XLNet on SQuAD v2.0 task*

|       Model       | Train Data | # Epoch | # Train Steps | Batch Size | Max Length | Learning Rate |    EM    |    F1    |
|:-----------------:|:----------:|:-------:|:-------------:|:----------:|:----------:|:-------------:|:--------:|:--------:|
|     XLNet-base    |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   80.23  |   82.90  |
|     XLNet-large   |  SQuAD 2.0 |    ~3   |     8,000     |     48     |    512     |      3e-5     |   85.72  |   88.36  |

*Table 2: The dev set performance of XLNet model finetuned on SQuAD v2.0 task*

## Reference
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. [SQuAD: 100,000+ questions for machine comprehension of text](https://arxiv.org/abs/1606.05250) [2016]
* Pranav Rajpurkar, Robin Jia, and Percy Liang. [Know what you don’t know: unanswerable questions for SQuAD](https://arxiv.org/abs/1806.03822) [2018]
* Danqi Chen. [Neural reading comprehension and beyond](https://cs.stanford.edu/~danqi/papers/thesis.pdf) [2018]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le and Ruslan Salakhutdinov. [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860) [2019]
