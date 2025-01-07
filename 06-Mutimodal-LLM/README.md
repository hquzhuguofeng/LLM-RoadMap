
 
# 06-Mutimodal-LLM

## 目录

- [上手指南](#上手指南)
  - [从零到一继续预训练LLaVA概述](#从零到一继续预训练LLaVA概述)
- [步骤](#步骤)
  - [多模态数据](#多模态数据)
  - [模型部分](#模型部分)
  - [具体教程](#具体教程)
  - [训练策略](#训练策略)
  - [训练技巧](#25-训练技巧)
- [文件目录说明](#文件目录说明)
- [版本控制](#版本控制)
- [作者](#作者)
- [版权说明](#版权说明)
- [鸣谢](#鸣谢)

## 上手指南

### 1-从零到一继续预训练LLaVA概述

1. 模型构建：基于`openai/clip-vit-large-patch14-336` 和`Qwen2.5-3B-Chat`模型，构建一个llava模型
2. 数据构建：`liuhaotian/LLaVA-CC3M-Pretrain-595K`
3. 训练方式：基于`deepspeed-zero2`，有`lora`训练、全量参数训练、冻结视觉层进行训练等方式。

文本编码：`text_encoder` 这是张[图]很好看<br>
图像编码：`picture_encoder` [图]<br>
融合：将`picture_encoder`的向量插入到`text_encoder`中，完成模型的编码和输出

### 2-步骤

#### 2.1-多模态数据


| 数据名称                     | 下载链接                                                                                                                                       | 数据质量                  | 数据量   |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|-------|
| TextOCR-GPT4o            | [https://huggingface.co/datasets/CaptionEmporium/TextOCR-GPT4o](https://huggingface.co/datasets/CaptionEmporium/TextOCR-GPT4o)             | 非常高👍                 | 2万条左右 |
| LLaVA-CC3M-Pretrain-595K | [https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) | 质量一般，但是具有代表性，可以用来做预训练 | 59万左右 |
| ShareGPT-4o              | [https://huggingface.co/datasets/OpenGVLab/ShareGPT-4o](https://huggingface.co/datasets/OpenGVLab/ShareGPT-4o)                             | 非常推荐👍                |       |


#### 2.2-模型部分
step1 增加词表 `<image>`

step2 用 **`openai/clip-vit-large-patch14-336`** 和 **`Qwen/Qwen2.5-7B-Chat`** 初始化llava模型

step3 补充token
- 补充llava模型`pad_token_id` 和 `image_token_index`
- llava文字部分的`pad_token`需要对齐qwen2.5的pad_token(<|endoftext|>) 
- 原本clip中`image_token_index` 变更为qwen2.5中的新增\<image>

step4 保存模型
- 保存model权重
- 保存模型的`processor`，其中包含llm的`tokenizer`和clip的处理器`autoprocessor`
- 🌟**主要需要把`show_model/model002`里面的`preprocessor_config.json`文件，放在`show_model/model001`里面**

step5 组织数据集
- 构建llavaDataset，读取单条数据
- 构建dataloader，目的是组织成batch的形式
- 输入给模型中

#### 2.3-具体教程

| 任务流程          | 细节 | 关联代码 |
|------------------|--------------------------------------------------------|---------|
| 从0到1构建llava模型 | 1. 如何从0到1构建一个空的属于自己的llava模型<br/>2. 加深对llava模型的认识，为训练模型做铺垫 | [01_build_model_show.ipynb](./01_build_model_show.ipynb)    | 
| 构建训练数据集 | 如何基于`liuhaotian/LLaVA-CC3M-Pretrain-595K`数据集，构建训练数据集      | [train_llava/train_llava/data.py](./train_llava/data.py)     | 
| 训练流程  | 1. 基于transformers框架，搭建训练代码<br/>2. 实现多重模式的训练。| [train_llava/run_zero2.sh](./run_zero2.sh)  |
| 推理   | 训练的模型，如何进行推理   | 1. lora版本： [code05_infer_lora.ipynb](./code05_infer_lora.ipynb) <br/>2. 全量参数版本:[train_llava/code05_infer.ipynb](./code05_infer.ipynb) |   

#### 2.4-训练策略

| 训练方式                         | 视觉层  | 转接层          | 语言层        | 效果评估（非常主观）                                                   |
|------------------------------|------|--------------|------------|--------------------------------------------------------------|
| `--train_type use_lora`      | 冻结🧊 | 随机初始化参数、冻结🧊 | 训练🔥（部分参数） | 效果非常好 👍（搞了一个bug：给转接层初始化了参数，但是没训练，效果也是很不错）😅（不建议这么做，但是可以试一试） |
| `--train_type use_lora`      | 冻结🧊 | 训练🔥         | 训练🔥（部分参数） | 效果非常好 👍                                                     |
| `--train_type none`          | 训练🔥 | 训练🔥         | 训练🔥       | 效果非常差👎                                                      |
| `--train_type freeze_vision` | 冻结🧊 | 训练🔥         | 训练🔥（全量参数） | 效果可以👍（比use_lora稍微差一点）                                       |

1. 训练的时候，使用lora方式进行训练最好。在`run_zero2.sh`里面设置`--train_type use_lora`即可。
2. 全量参数训练，效果非常差。
3. 上面说到的【效果评估】、都是在特定数据的情况下，在自己的数据上，建议都试一试，不是绝对的结论。
4. 转接层指的是lora_config中的`multi_modal_projector`参数,指定了则开启了训练

#### 2.5-训练技巧

为了可以异步的处理数据，可以在`run_zero2.sh`里面使用这三个参数

```shell
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \

```

基本上可以提高1倍的训练效率。
参考链接：

1. https://developer.aliyun.com/article/914214
2. https://blog.csdn.net/qq_32527569/article/details/134777374

### 3-文件目录说明
eg:

```
06-Multimodal-LLM
├── train_llava
│   ├── data.py
│   └── util.py
├── 01_build_model_show.ipynb
├── code05_infer_lora.ipynb
├── code05_infer.ipynb
├── ds_zero2_no_offload.json
├── model_llava.py
├── README.md
├── run_zero2.sh
└── run.py
```



### 4-版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 5-作者

[GuoFeng Github](https://github.com/hquzhuguofeng)

[GuoFeng CSDN](https://blog.csdn.net/weixin_46133588?spm=1011.2415.3001.5343)

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 6-版权说明

None

### 鸣谢


- [Transformers lessons](https://github.com/zyds/transformers-code)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/v4.27.2/zh/index)
- [llava paper](https://github.com/haotian-liu/LLaVA)
- [llava lessons](https://space.bilibili.com/45156039/channel/series)



