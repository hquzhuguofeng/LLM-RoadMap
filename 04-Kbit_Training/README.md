
 
## 目录

- [bitfit](#bifit)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)

1. 数据集：https://huggingface.co/datasets/shibing624/alpaca_data_zh
2. 预训练模型：Langboat/bloom-1b4-zh

### 01-低精度训练和模型下载

模型训练的显存占用
- 模型权重（4Bytes * 模型参数量）
- 优化器状态 （8Bytes * 模型参数量）对于常规的AdamW而言
- 梯度 （4Bytes * 模型参数量）
- 前向激活值 （取决于序列长度、隐层维度，Batch大小）

降低训练显存的技巧
- 梯度累计
- 梯度检查点
- 优化器配置，梯度占用小的优化器    
- 输入数据长度
- 冻结模型参数
参数高效微调
- prompt-tuning/lora

如何降低参数占用的字节数
- 默认的数值精度为单精度fp32，32bit, 4字节
- 1b参数量大概是4G显存
- 常见的低精度数据类型：fp16\bf16\int8\int4




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


