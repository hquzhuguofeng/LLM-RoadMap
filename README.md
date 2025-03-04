<div align="center">

<img src="./assets/LLM_roadmap.png" width="300em" ></img> 


<span style="display: inline-flex; align-items: center; margin-right: 2px;">
   <a href="https://blog.csdn.net/weixin_46133588?spm=1011.2415.3001.5343" target="_blank"> CSDN</a> &nbsp;|
 </span>
  <span style="display: inline-flex; align-items: center; margin-left: 2px;">
   LLM_RoadMap <a href="README.md" target="_blank">&nbsp; 📖 最佳实践</a>
 </span>

</div>


---

## 概述

1. 🎯`目标`：基于`transformers`、`pytorch`等内容实现了多种NLP任务，内容不限于：Transformers数据处理、传统NLP任务、模型训练（包含LLM、embedding、VL-Model）；
2. 💽`数据`：
    - 整理经典NLP任务数据集，帮助用户上手
    - 梳理原理，讲解接口，用户可根据需求自定义数据集，进行任务的构建
3. 💻`流程`：该项目核心基于transformers库构建NLP中常规的任务，你可以根据需求更新对应的数据集；
    - **Transformers实战**，通过详细的案例讲解transformers在传统NLP任务中数据流动、接口定义以及核心原理，掌握调用的基本原理，内容不限于：NER、MRC、多项选择题、文本相似度、检索式问答机器人、掩码语言模型、生成式对话机器人
    - **高效微调**，以PEFT库为核心，讲解常见的参数高效微调的原理和实战，不限于：Bitfit、Prompt-Tuning、Ptuning、Prefix_tuning、Lora、ia3、自定义模型
    - **低精度模型训练**，以bitsandbytes库，进行模型的低精度训练，包括半精度、8bit训练、4bit训练（QLora）
    - **分布式模型训练**，基于accelerate库，进行模型分布式训练，介绍分布式训练的基本原理，accelerate库的基本使用，包含deepspeed框架的集成
4. 🔥`模型`：当前已经支持`llava`多模态大模型, 基于`clip`和`qwen2.5`从零训练一个llava模型；

## 指引

| 中文名称 | 文件夹名称| 数据 | 库接口讲解 | 大模型 | 模型部署 | 
|---------|----------|------|---------|--------|---------|
| Transformers实战 | [transformers_practices](./01-transformers_practices/README.md) | ✅  | ✅    | ✅   | ❌    | 
| NLP_tasks实战 | [NLP_tasks](./02-NLP_tasks/README.md) | ✅  | ✅  | ✅  | ✅ | 
| 模型微调实战 | [PEFT](./03-PEFT/README.md) | ✅  | ✅  | ✅  | ✅ | 
| 模型量化微调实战 | [Kbit_Traning](./04-Kbit_Training/README.md) | ✅  | ✅  | ✅  | ✅ | 
| 模型分布式训练实战 | [Distrbution_Training](./05-Distrbution_Training/README.md) | ✅  | ✅  | ✅  | ✅ | 
| 从零开始训练多模态模型 | [Multimodal-LLM](./06-Mutimodal-LLM/README.md) | ✅  | ✅  | ✅  | ✅ | 

## 更新日志

#### 📌 置顶
* [2025.03.04] 🍬基于self-instrut基本原理实现`domain_datasets`的自我扩充, 详情参考[self-intruct-zh.md](./02-NLP_tasks/self-instruct-zh/README.md)
* [2025.02.20] 🌟🌟🌟基于`clip`和`qwen2.5`模型，基于`deepspeed-zero2`从零开始训练`llava`模型
* [2025.01.21] 🚀MOE block的实际运行机理和数据流动室验，[moe_block_demo](./02-NLP_tasks/15-moe_block_demo.py)；参考`liger_kernel`的内容，基于`triton`实现`MLP`加速
* [2025.01.20] 💪基于accelerate和deepspeed框架，实现`Data Parrallel`、`Distributed Data Parrallel`、zero2、zero3策略的模型训练
* [2025.01.15] 🌀低精度训练框架bitsandbytes库实现半精度、8bit训练、qlora训练
* [2024.12.30] 🔥基于高效微调框架PEFT实现`bitfit`、`prompt-tuning`、`Ptuning`、`prefix-tuning`、`lora`、`ia3`和`自定义模型`
* [2024.12.22] 💫 更新检索机器人、生成式问答机器人，文本摘要的具体实现
* [2024.12.20] 💫 更新基于transformers的NLP任务的构建，包含命名实体识别、阅读理解（滑窗机制）、多项选择题、文本相似计算（单塔和多塔模型）、UIE信息抽取
    > 模型包含：pointwise、DSSM、sentence_transformer、simcse、sbert等训练和推理


<details> 
<summary>点击查看完整更新日志。</summary>

* [2024.12.13] ⭐️⭐️⭐️ Transformers_practices仓库更新，包含`pipeline`、`tokenizer`、`Model`加载与保存、模型训练流程搭建(`Datasets`、`Evaluate`、`Trainer`)

</details>


## 作者

[GuoFeng Github](https://github.com/hquzhuguofeng)<br>
[GuoFeng CSDN](https://blog.csdn.net/weixin_46133588?spm=1011.2415.3001.5343)


## 鸣谢


- [Transformers lessons-1](https://github.com/zyds/transformers-code)
- [Transformers lessons-2](https://github.com/HarderThenHarder/transformers_tasks.git)
- [Transformers lessons-3](https://github.com/yuanzhoulvpi2017/zero_nlp)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/v4.27.2/zh/index)
- [Pytorch]()
