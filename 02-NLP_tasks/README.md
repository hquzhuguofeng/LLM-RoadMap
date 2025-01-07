
 
# 02-NLP_tasks

## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
  - [文本匹配](#文本匹配任务)
  - [信息抽取](#信息抽取)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版本控制](#版本控制)
- [作者](#作者)
- [版权说明](#版权说明)
- [鸣谢](#鸣谢)

## 上手指南

### 开发前的配置要求

1. Python == 3.11.8
2. PyTorch == 2.1.1+cu121
3. transformers == 4.47.1

### 安装步骤


```sh
git clone https://github.com/hquzhuguofeng/LLM-RoadMap.git
```

### 文件目录说明
eg:

```
02-NLP_tasks
├── __pycache__
├── data
├── dual_model
├── masked_lm
├── model_for_ner
├── multiple_choice
├── UIE_torch
├── 00-template.ipynb
├── 01-transformer_nlp.ipynb
├── 02-ner.ipynb
├── 03-mrc_simple.ipynb
├── 04-mrc_slice_windows.ipynb
├── 05-multiple_choice.ipynb
├── 06-cross_model.ipynb
├── 07-dual_model.ipynb
├── 07-01-inference_pointwise.py
├── 07-01-train_pointwise.py
├── 07-02-inference_dssm.py
├── 07-02-train_dssm.py
├── 07-03-inference_sentence_transformer.py
├── 07-03-train_sentence_transformer.py
├── 07-04-inference_simcse.py
└── 07-04-train_simcse.py
├── 08-retrieval_bot.ipynb
├── 09-masked_lm.ipynb
├── 10-causal_lm.ipynb
├── 11-summarization_t5.ipynb
├── 12-summarization_glm.ipynb
├── 13-chatbot.ipynb
├── cmrc_eval.py
├── dual_model_bert.py
├── get_embedding.py
├── iTrainingLogger.py
├── metric_accuracy.py
├── metric_f1.py
├── model.py
├── mrc_slice_windows.py
├── README.md
├── seqeval_metric.py
├── TEXT_MATCHING.md
└── utils.py
```

#### 文本匹配任务
对应文件：
- 06-cross_model.ipynb
- 07-dual_model.ipynb

- 07-01-train_pointwise.py 单塔模型
```
python train_pointwise.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "ERNIE-PointWise" \
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```
- 07-01-inference_pointwise.py


- 07-02-train_dssm.py 双塔模型 DSSM
```
python train_dssm.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/dssm" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "ERNIE-DSSM" \
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```
- 07-02-inference_dssm.py

- 07-03-train_sentence_transformer.py 双塔模型 sentence_transformers
```
python train_sentence_transformer.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/sentence_transformer" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "Sentence-Ernie" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:1"
```
- 07-03-inference_sentence_transformer.py

- 07-04-train_simcse.py
```
python train_simcse.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/LCQMC/train.txt" \
    --dev_path "data/LCQMC/dev.tsv" \
    --save_dir "checkpoints/LCQMC" \
    --img_log_dir "logs/LCQMC" \
    --img_log_name "ERNIE-ESimCSE" \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --batch_size 64 \
    --max_seq_len 64 \
    --valid_steps 400 \
    --logging_steps 50 \
    --num_train_epochs 8 \
    --device "cuda:2"
```
- 07-04-inference_simcse.py

详情查看：[Text Matching MD](TEXT_MATCHING.md)

#### 信息抽取
对应文件：
- 02-ner.ipynb
- UIE_torch

详情查看：[UIE MD](./UIE_torch/readme.md)

### 使用到的框架

- pytorch
- transformers

### 贡献者

GuoFeng


### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

[GuoFeng Github](https://github.com/hquzhuguofeng)

[GuoFeng CSDN](https://blog.csdn.net/weixin_46133588?spm=1011.2415.3001.5343)

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

None

### 鸣谢


- [Transformers lessons](https://github.com/zyds/transformers-code)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/v4.27.2/zh/index)



