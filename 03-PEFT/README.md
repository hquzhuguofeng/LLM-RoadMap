
# 02-NLP_tasks

## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
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
4. peft == 0.11.1

### 安装步骤


```sh
git clone https://github.com/hquzhuguofeng/LLM-RoadMap.git
```

1. 数据集：https://huggingface.co/datasets/shibing624/alpaca_data_zh
2. 预训练模型：Langboat/bloom-1b4-zh

### 文件目录说明
eg:

```
03-PEFT
├── chatbot
├── chatbot_prefix_tuning
├── chatbot_ptuning
├── data
├── 01-chatbot_bitfit.ipynb
├── 02-chatbot_prompt_tuning.ipynb
├── 03-chatbot_ptuning.ipynb
├── 04-chatbot_prefix_tuning.ipynb
├── 05-chatbot_lora.ipynb
├── 06-chatbot_ia3.ipynb
└── 07-peft_advanced_operations.ipynb
```

#### 01-chatbot_bitfit

使用bitfit微调参数

  可训练参数占模型的 0.000418051659240749

#### 02-chatbot_prompt_tuning

使用prompt_tuning微调参数, 有两种方式，一种是硬编码，一种是软编码，区别是初始化的方式不同


  trainable params: 14,336 || all params: 1,303,126,016 || trainable%: 0.0011

#### 03-chatbot_p_tuning

使用prompt_tuning微调参数, 针对prompt tuning的soft进行优化，在embedding层的soft中增加一个MLP or LSTM增加收敛速度

  重参数层是MLP
  trainable params: 12,609,536 || all params: 1,315,721,216 || trainable%: 0.9584

  重参数层是LSTM
  trainable params: 193,030,144 || all params: 1,496,141,824 || trainable%: 12.9019

  明显LSTM可训练的参数更多。

#### 04-chatbot_prefix_tuning

使用prefix_tuning微调参数, 之前的prompt_tuning、p_tuning都是只针对embedding层进行微调，prefix_tuning是将一部分参数放到模型的每层中进行。

  如果是prefix_projection = False，仅在embedding上新的参数。
  trainable params: 983,040 || all params: 1,304,094,720 || trainable%: 0.0754

  如果是prefix_projection = True, 在大模型的embedding和每层前都加上新的参数。
  trainable params: 205,641,728 || all params: 1,508,753,408 || trainable%: 13.6299

#### 05-chatbot_lora

使用lora tuning微调参数, 核心是作者认为模型训练存在一个内在维度，单独训练这个内在维度，好处是相比于p-tuning prompt-tuning prefix-tuning没有额外的参数量

  - transformers == 4.30.0
  - peft == 0.11.1
  - accelerate == 0.32.1

  设置的参数如下：
  - config = LoraConfig(task_type=TaskType.CAUSAL_LM)
  默认参数： lora的秩r;target_modules进行微调的对象；lora_alpha缩放因子；modules_to_save还可以对模型其他地方进行微调
  LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='D:/pretrained_model/models--Langboat--bloom-1b4-zh', revision=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, r=8, target_modules={'query_key_value'}, lora_alpha=8, lora_dropout=0.0, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, layer_replication=None)

  trainable params: 1,572,864 || all params: 1,304,684,544 || trainable%: 0.1206

  如果参数设置如下：
  - config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["query_key_value", "dense_4h_to_h"])
  则训练参数为：
  trainable params: 3,538,944 || all params: 1,306,650,624 || trainable%: 0.2708

  - config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"])
  trainable params: 95,225,856 || all params: 1,398,337,536 || trainable%: 6.8099

#### 06-chatbot_ia3
  - 核心思想是通过可学习的向量对激活值抑制或放大。具体来说，会对k\v\ffn三部分进行调整，训练过程中同样冻结原始模型权重，只更新可学习的部分向量部分。
  - config = IA3Config(task_type=TaskType.CAUSAL_LM)
  trainable params: 344,064 || all params: 1,303,455,744 || trainable%: 0.0264

#### 07-peft_advanced_operation
  - 如何peft微调自定义模型，直接通过target_model设置
  - 一个主模型，多个适配器，用set_adapter进行配置，主要用于多任务的模型微调，不同的模型适配不同的lora，后期根据任务不同，走不同的Lora配置。
  - 如何获取原始模型的输出结果（禁止适配器）进行原模型的对比或者说在训练DPO的时候，是需要原始模型的输出。


### 使用到的框架

- pytorch
- transformers
- peft

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
