
 
## 目录

- [bitfit](#bifit)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)



### 01-分布式训练基础与环境配置

单卡场景下解决显卡问题
- 可训练参数【lora\p-tuning】
- 低精度参数训练【float16\bfloat16\int8\qlora】

分布式训练训练
- 原理：分布式(Distributed)是指系统或计算任务被分布到多个独立的节点或计算资源上进行处理，而不是集中在单个节点或计算机上。分布式模型训练是一种机器学习和深度学习领域中的训练方法，它通过将计算任务分发到多个计算资源或设备上来加速模型的训练过程。分布式训练通过并行计算的方式，将数据和计算任务分配到多个节点上，从而提高训练速度和处理大规模数据的能力。
- 类别：
1. 数据并行-DP
每个GPU上训练的数据不同，每个GPU都复制一份完整的参数。<br>
**模型并行**
2. 流水并行-PP
将模型按层拆开，每个GPU上包含部分的层，保证都能够正常训练，不要求每张卡内都可以完整执行训练过程
3. 张量并行-TP
将模型按列拆开，每个GPU上包含部分的层，保证都能够正常训练，不要求每张卡内都可以完整执行训练过程
4. 混合策略
数据并行+流水并行+张量并行（3D并行）



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


