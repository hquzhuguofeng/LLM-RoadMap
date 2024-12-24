
 
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

### 02-Data Parrallel
Data Parrallel 原理
- 训练流程
- Step1 GPU0 加载model和batch数据
- Step2 将batch数据从GPU0均分至各卡
- Step3 将model从GPU0复制到各卡
- Step4 各卡同时进行前向传播
- Step5 GPU0收集各卡上的输出，并计算Loss
- Step6 将Loss分发至各卡，进行反向传播，计算梯度
- Step7 GPU0收集各卡的梯度，进行汇总
- Step8 GPU0更新模型

单GPU训练
- 只要指定torch.nn.DataParallel(model)然后loss设置为标量，即可完成train流程的搭建，其实不用加这行代码，默认trainer检测到有多卡也会默认启动DP进行训练

DP训练的问题和主要用途
- batch越大，节约的时间越多，不然相比于GPU训练，时间都浪费在通信上
因此，DP的问题，单进程，多线程，由于GIL锁的问题，不能充分发挥多卡的优势；由于DP的策略问题，会存在一个主节点占用比其他节点多的情况；效率较低，每次训练开始都要重新同步模型，大模型的同步时间会较难接受；只适用于单机训练，无法支持真正的分布式多节点训练。
- 对于并行推理，DataParallel可以排上用场
  - DataParallel.module.forward()
  - DataParallel.forward()
  - DataParallel.forward() 改进版本，把模型并行推理的代码摘出来，将模型复制的代码，单独拎出来

### 03-Distributed Data Parrallel
- 训练流程
- Step1 多进程，每个进程都加载数据和模型
- Step2 各进程同事进行前向传播，得到输出logit
- Step3 各进程分别计算Loss,反向传播，计算梯度
- Step4 各进程间通信，将梯度在各卡同步 gradient all-reduce
- Step5 各进程分别更新模型
分布式的基本概念
- group:进程组，一个分布式任务对应一个进程组，一般就是所有卡都在一个组里
- world size:全局的并行数，一般情况下等于总的卡数
- node:节点，可以是一台机器，或者一个容器，节点内包含多个GPU
- rank(global rank):整个分布式训练任务内的进程序号
- local rank:每个node内部的相对进程序号
分布式训练中的通信
- 在分布式模型训练中，通信是不同计算节点之间进行信息交换以协调训练任务的关键组成部分。
通信类型
- 点对点通信:将数据从一个进程传输到另一个进程称为点对点通信
- 集合通信:一个分组中所有进程的通信模式称之为集合通信全
  - 6种通信类型:Scatter、Gather、Reduce、All Reduce、Broadcast、All Gather
  - scatter是分发；一个主进程分发到多个进程中
  - gather是收集；多个进程的信息汇聚到一个主进程中
  - reduce是在scatter的基础上将数据进行计算【加减乘除】多个进程汇总到一个进程上；
  - all-reduce 多进程的信息在多进程间同步聚合+运算
  - broadcast 就是一个数据广播到其他进程中
  - all-gather 多进程的信息在多进程间同步聚合





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


