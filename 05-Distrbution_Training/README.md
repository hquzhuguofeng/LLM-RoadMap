
 
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
代码实战
- 定义进程组nccl
- dataloader中的collate_fn需要自己去定义
- DistributedDataParallel去定义模型
- 模型初始化中需要放到0号节点上；同时模型训练和推理中，也是需要将数据放到对应rank的节点上；model.to(int(os.environ["LOCAL_RANK"]))；batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
- torchrun --nproc_per_node=2 ddp.py  nproc_per_node 几张卡   
- 因为每个进程都是单独的数据和模型，需要自己实现通信的内容
  - train_loss部分需要all_reduce每个进程的loss并进行平均值的计算
  - 打印loss, 用gobal_rank的内容进行控制打印loss
  - eval_acc，需要all-reduce每个进程的acc的内容
- 额外小细节：
  - 在 DDP 训练中，虽然不同的进程会处理不同的数据子集，但通过正确的配置和工具（如 DistributedSampler 和随机数生成器），可以确保这些数据子集的划分是一致的，从而保证训练过程的稳定性和可复现性
  - trainset, validset = random_split(dataset, lengths=[train_length, valid_length], generator=torch.Generator().manual_seed(42)) 通过指定generator来解决，数据切分的稳定性
  - 在 DDP 训练中，调用 trainloader.sampler.set_epoch(ep) 是非常重要的，它可以确保数据在每个 epoch 中被打乱并且所有进程处理的数据子集是一致的。数据打乱、跨进程一致性、避免重复数据
huggingface trainer自带DDP

**总结**

数据部分
- 如果是在训练进程内对数据集进行划分，注意保证数据划分的一致性，可以通过随机种子控制分布式采样器会为了保证每个进程内数据大小一致，做额外的填充，评估指标可能会存在误差

书写逻辑
- 将分布式的代码看作单进程的代码即可，只是需要分布式的数据采样器以及启动略有不同print打印的都是各自进程内的信息，需要全局的信息则需要自行调用通信计算结果数据放置到指定设备上时需要注意使用正确的deviceid，一般用local_rank


### 04-Accelerate分布式训练
介绍
- Accelerate是Huggingface生态中针对分布式训练推理提供的库，目标是简化分布式训练的流程
- Accelerate库本身不提供分布式训练的内容，但是其内部集成了多种分布式训练框架DDP、FSDP、Deepspeed等
- Accelerate库提供了统一的接口，一套代码搞定多种分布式训练框架，简单的几行代码(4行)便可让单机训练的程序变为分布式训练程序
- Transformers库中也是通过Accelerate集成的分布式训练，因此Accelerate库的学习非常有必要!
相比上一个版本的改动如下：
- 数据处理部分：不用指定 DistributedSampler
- 模型部分：不用DDP进行包装了
- 模型评估部分：使用gather_for_metrics进行数据的分析
- 包括模型训练部分的内容，accelerate同样提供了reduce loss的方法
- 训练的脚本：torchrun --nproc_per_node 2 05-ddp_accelerator.py
- accelerate同样提供了启动的命令：accelerate launch 05-ddp_accelerator.py
- accelerate config 可以配置分布式模型训练的参数
  - This machine\multi-GPU\1 multi-node\no:dynamo、deepspeed、FSDP、Megatorn-LM\2GPUs\no mixed_precision

accelerate使用进阶
- 混合精度训练
  - 前向传播中使用 FP16，在关键操作（如梯度计算和参数更新）中使用 FP32，混合精度训练可以在不牺牲太多准确性的情况下显著提高训练效率
  - 混合精度训练中的显存占用
    - 假设模型参数量为M

    | 项目       | 混合精度训练           | 单精度训练         |
    |------------|------------------------|--------------------|
    | 模型       | (4+2) Bytes * M        | 4 Bytes * M        |
    | 优化器     | 8 Bytes * M            | 8 Bytes * M        |
    | 梯度       | (2 + ) Bytes * M       | 4 Bytes * M        |
    | 激活值       | 2 Bytes * A          | 4 Bytes * A        |
    | 汇总       | (16 + ) Bytes * M +2 Bytes * A     | 16 Bytes * M  + 4 Bytes * A     |
    - **混合精度训练可以加速训练，但不一定会降低模型显存占用**
    - **激活值占用比较大时，可以看到明显的显存占用降低**
  - 启动方式
    - 方式一 accelerator = Accelerator(mixed precision="bf16") # 当前机子不支持bf16,切换为fp16可运行成功
    - 方式二 acclerator config && choice bf16
    - 方式三 accelerator launch --mixed precision bf16 {script.py}
  - apt-get install nvtop 监测显存占用
- 梯度累积
  - 分割Batch:将大的训练数据Batch分割成多个小的Mini-Batch。
  - 计算梯度:对每个Mini-Batch独立进行前向和反向传播，计算梯度。
  - 累积梯度:不立即更新模型参数，而是将这些小Batch的梯度累积起来
  - 更新参数:当累积到一定数量的梯度后，再统一使用这些累积的梯度来更新模型参数
  - accelerator=Accelerator(gradient_accumulation_steps=xx)
  - 训练过程中，加入accelerator.accumulate(model)的上下文
  - with accelerator.accumulate(model):
- 实验记录工具
  - Tensorboard
    - Accelerate 日志记录功能
    - 步骤一：
      - 创建Accelerator时指定project_dir
      - accelerator = Accelerator(log with="tensorboard", project dir="xx")
    - 步骤二:
      - 始化tracker
      - accelerator.init trackers(project name="xx")
    - 步骤三:
      - 结束训练，确保所有tracker结束
      - accelerator.end_training()
  - WandB
  - CometML
  - Aim
  - MLflow
  - Neptune
  - Visdom
- 模型保存点
- 断点续训





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


