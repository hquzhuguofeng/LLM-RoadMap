
 
## 目录

- [bitfit](#bifit)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)

1. 数据集：https://huggingface.co/datasets/shibing624/alpaca_data_zh
2. 预训练模型：Langboat/bloom-1b4-zh

### 01-chatbot_bitfit

使用bitfit微调参数

  可训练参数占模型的 0.000418051659240749

### 02-chatbot_prompt_tuning

使用prompt_tuning微调参数, 有两种方式，一种是硬编码，一种是软编码，区别是初始化的方式不同


  trainable params: 14,336 || all params: 1,303,126,016 || trainable%: 0.0011

### 03-chatbot_p_tuning

使用prompt_tuning微调参数, 针对prompt tuning的soft进行优化，在embedding层的soft中增加一个MLP or LSTM增加收敛速度

  重参数层是MLP
  trainable params: 12,609,536 || all params: 1,315,721,216 || trainable%: 0.9584

  重参数层是LSTM
  trainable params: 193,030,144 || all params: 1,496,141,824 || trainable%: 12.9019

  明显LSTM可训练的参数更多。

### 04-chatbot_prefix_tuning

使用prefix_tuning微调参数, 之前的prompt_tuning、p_tuning都是只针对embedding层进行微调，prefix_tuning是将一部分参数放到模型的每层中进行。

  如果是prefix_projection = False，仅在embedding上新的参数。
  trainable params: 983,040 || all params: 1,304,094,720 || trainable%: 0.0754

  如果是prefix_projection = True, 在大模型的embedding和每层前都加上新的参数。
  trainable params: 205,641,728 || all params: 1,508,753,408 || trainable%: 13.6299

### 05-chatbot_lora

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


### 文件目录说明
eg:

```
filetree 
├── ARCHITECTURE.md
├── LICENSE.txt
├── README.md
├── /account/
├── /bbs/
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── manage.py
├── /oa/
├── /static/
├── /templates/
├── useless.md
└── /util/

```





### 开发的架构 

请阅读[ARCHITECTURE.md](https://github.com/shaojintian/Best_README_template/blob/master/ARCHITECTURE.md) 查阅为该项目的架构。

### 部署

暂无

### 使用到的框架

- [xxxxxxx](https://getbootstrap.com)
- [xxxxxxx](https://jquery.com)
- [xxxxxxx](https://laravel.com)

### 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。


### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

xxx@xxxx

知乎:xxxx  &ensp; qq:xxxxxx    

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*


