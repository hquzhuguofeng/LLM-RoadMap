
 
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

### 02-llama2_lora_16bit

transformers == 4.38.0
accelerate == 1.2.1

模型用的是：Llama-3.2-1B

如果模型使用batch_size训练，padding部分没有处理则会报错，因此需要指定tokenizer.pad_token = tokenizer.eos_token
- 分词时候，指定最大长度 MAX_LENGTH = 1024
- 分词时候，需要指定padding_token_id 为 eos_token_id, 原因llama模型分词器会将非单独存在的eos token切开，因此对于eos token要单独处理，否则训练的模型在预测时候不知道怎么停止
- 半精度训练时，正确加入eos_token后，要将padding token id 也要置为 eos token id 否则模型无法收敛，这种情况是在float16 batch_size为2的情况下才会出现
- lora: 用的是默认参数
- batch_size: 2
- model 使用bfloat16进行加载
- trainable params: 851,968 || all params: 1,236,666,368 || trainable%: 0.0689
- 显存占用6.8G左右

在以上配置的情况下，打开gradient_checkpointing=True，会报错，需要打开 model.enable_input_require_grads()
- 显存占用到了6G左右

如果模型加载的时候变成float16情况下，loss发散了，无法收敛，原因是参数溢出问题,
- 调整adam的adam_epsilon的上限即可
- 原理是torch.tensor(1e-8).half() -> tensor(0., dtype=torch.float16) 上溢出了
- adam_epsilon=1e-5

### 03-chatglm3_lora_16bit
GLM架构是集合了自回归和自编码架构的特点。
prefix部分和target部分的内容。
- x1 x2 M x4 M   进行span的预测
- x3 /  x5  x6
第二点需要说明的是
- 位置编码有两重
- 第一重的位置编码 target在原本句子的位置
- 第二重的位置编码 target的内在顺序
- prefix是full attention
- target是prefix+self decoder attention

chatglm模型讲解
- 模型版本有v1和v2，这两个版本的区别是
- 训练方式上，v1 prefix LM ; v2是 causal LM  在训练多轮的情况下v2的效率更高点；v3更多样的训练数据，更充分的训练步数，更合理的训练策略
- 数据格式，[Round N]\n\n问：Prompt \n\n答：Response
- 数据组织 v1是prefix learning 的组织方式，prefix部分是full attention
- v2是将prompt放在后面了，
- v3 开放了base模型，支持工具调用，支持代码执行
- chatglm3新版本特性是对prompt进行一个改造

```
<|system|>
you are chatglm3, a large language model.
<|user|>
Hi
<|assistant|>
hello，i am chatglm3.
<|observation|>
```
- chatglm3相当于有自己定义的prompt的格式，因此数据在处理的过程中，需要满足其要求,可以通过?model.chat方法查看调用的模板
- special token需要单独处理，包括角色类（user\assistant等）和eos_token
- 对于非工具调用功能和代码执行功能的训练，要求解码的第一个token一定是"\n"
- chatglm3已经写好了prompt的定义的方法tokenizer.build_chat_input的方法； 不过拿到的是tensor，需要转化为list的格式
- 如果使用lora训练最好指定Tasktype, 不然要指定remove_unused_columns为False，否则会报很奇怪的错误，很难debug

### 04-8bit的量化
量化就是将高精度的数字通过某种手段将其转换为低精度的数据。量化正常来说就是量化模型权重。
`int8量化`（即将数据从浮动精度转换为8位整数精度）是一种常用于深度学习模型优化的技术，它通过减少数值表示的精度来加速计算和减小模型存储需求，同时尽可能减少模型精度损失。量化方法通常有多种，其中 `absmax` 是一种常见的量化方法。

`absmax`量化方法是通过找到原始数据（通常是浮点数）的最大绝对值来确定量化的比例因子。然后，这个比例因子用于将数据映射到一个较小的整数范围（通常是-128到127），从而实现整数表示。具体步骤如下：

1. **找出最大绝对值（absmax）**：首先计算所有值的绝对值，然后选择其中最大的绝对值，记为 `max_value`。
2. **确定量化比例因子**：根据最大绝对值来确定一个比例因子，将浮点数值映射到整数值的范围。例如，使用 `max_value` 和目标整数范围的最大值（例如，对于int8为127）来计算比例因子。
3. **应用比例因子进行量化**：通过除以比例因子，将浮动数值转换为整数值。


假设我们有一个模型的权重数组如下：
```
[0.1, -0.3, 0.25, 0.15, -0.05]
```

我们将使用 `absmax` 量化方法对其进行int8量化。

1. **找出最大绝对值（absmax）**：
   - 原数组的最大绝对值是 `max(abs(0.1), abs(-0.3), abs(0.25), abs(0.15), abs(-0.05))`，即 `0.3`。

2. **确定量化比例因子**：
   - 对于int8，目标范围是从 -128 到 127。我们可以选择比例因子为 `max_value / 127`，也就是：0.3/127=0.00236

3. **量化浮点数为整数**：
   - 将每个浮动值除以比例因子，然后四舍五入到最接近的整数,量化后的int8数组是：
     ```
     [42, -127, 106, 64, -21]
     ```
4. **反量化**：
    [42, -127, 106, 64, -21] * 量化因子0.00236 = [0.099, -0.3, 0.250, 0.151, -0.05]

通过这种方式，我们将原始的浮动数组转换为8位整数数组，从而减少了模型的存储需求，并加速了后续的计算。但是也有缺陷，**离群值**

- 混合精度分解量化LLM.int8()
如果是正常值走常规的absmax量化，如果是异常值走的是半精度的量化，然后两部分的内容加起来。
  - 从输入的隐含状态中，按列提取离群值，即大于某个阈值的值
  - 对于FP16离群值矩阵和INT8非离群值矩阵分别作矩阵乘法
  - 反量化非离群值的矩阵乘结果 + 离群值矩阵乘结果，最终获得FP16结果。

环境安装<br>
bitsandbytes github 下载windows 64编译好的文件，进行pip install
- 使用，load_in_8bit=True即可
- 注意，如果训练了一个8bit的模型，最好不要和原模型进行合并，会出现精度round的问题

### 05-4bit的量化（qlora量化）





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


