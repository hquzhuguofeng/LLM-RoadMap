
# Transformers库的学习

本项目展示了如何使用[Hugging Face Transformers](https://github.com/huggingface/transformers)库中的`pipeline`函数、`tokenizer`函数

## 01-pipeline 基本使用指南

1. **查看支持的任务类型**

2. **创建和使用Pipeline**
   - 使用默认模型创建Pipeline
   - 指定模型创建Pipeline
   - 预先加载模型创建Pipeline
   - 使用GPU加速推理

3. **确定Pipeline参数**

4. **其他Pipeline示例**
   - 零样本对象检测
   
5. **Pipeline背后的实现**


## 02-tokenizer 基本使用指南


1. **简介**
   - 什么是Tokenizer？
   - Tokenizer的作用

2. **加载与保存**
   - 从Hugging Face加载分词器
   - 分词器保存到本地
   - 从本地加载分词器

3. **句子分词**
   - 使用`tokenize`方法进行分词
   - 使用`encode`方法将句子转换为ID序列
   - 使用`decode`方法将ID序列转换回字符串

4. **查看词典**
   - 获取词典
   - 获取词典大小

5. **索引转换**
   - 将词序列转换为ID序列
   - 将ID序列转换为词序列
   - 将词序列转换为字符串

6. **填充与截断**
   - 填充至固定长度
   - 截断至固定长度

7. **其他输入部分**
   - 构建注意力掩码
   - 构建段落类型ID

8. **快速调用方式**
   - 使用`encode_plus`方法一次性完成多项操作
   - 直接调用分词器进行编码

9. **处理批量数据**
   - 对多个句子进行编码
   - 性能比较：单句循环处理 vs. 批量处理

10. **Fast vs. Slow Tokenizer**
    - 加载Fast Tokenizer
    - 加载Slow Tokenizer
    - 性能比较：Fast Tokenizer vs. Slow Tokenizer

11. **特殊Tokenizer的加载**
    - 加载特殊模型的分词器
    - 保存特殊分词器到本地
    - 从本地加载特殊分词器
    - 测试特殊分词器的编码和解码功能

 ## 03-使用Transformers库进行模型加载与保存


### 01-在线与离线模型加载指南

1. **在线加载**
   - 使用`AutoModel`类直接从Hugging Face加载预训练模型。
   - 强制重新下载模型以确保获取最新版本。

2. **模型下载**
   - 当需要科学上网时，通过Git或Git LFS手动克隆Hugging Face上的模型仓库。

3. **离线加载**
   - 从本地路径加载已经下载的预训练模型，适用于没有网络连接的情况。

4. **模型加载参数**
   - 加载模型配置信息，查看或修改模型参数，如输出注意力等设置。

### 02-模型调用指南

1. **简介**
   - 模型调用的基础知识，包括准备输入数据、选择模型类型等。

2. **不带Model Head的模型调用**
   - 调用基础模型进行预测，不附加特定任务头部（head）。
   - 设置模型参数，如启用输出注意力机制。

3. **带Model Head的模型调用**
   - 加载带有特定任务头部（例如分类、回归）的模型。
   - 对输入数据进行分类预测，并检查模型配置中的标签数量。

## 04-使用Transformers库进行文本分类

1. **数据加载与清理**
   - 使用Pandas读取CSV文件并清理掉含有缺失值的数据条目，确保数据集的完整性和一致性。

2. **数据集构建与划分**

   - 自定义了一个继承自`torch.utils.data.Dataset`的类，以适配特定的数据结构，并实现了`__getitem__`和`__len__`方法。
   - 通过`random_split`函数将数据集划分为训练集和验证集，以便后续评估模型性能。

3. **Dataloader创建**
   - 为了高效地批量处理数据，我们使用`DataLoader`结合自定义的`collate_func`函数，该函数负责对每个batch内的数据进行tokenization以及标签转换。

4. **模型加载与优化器配置**
   - 采用`AutoModelForSequenceClassification`从本地路径加载预训练的中文BERT模型（或任何支持的模型）。
   - 选择了AdamW作为优化算法，并设置了适当的学习率。

5. **训练与验证流程**
   - 定义了训练和验证函数，其中包含前向传播、损失计算、反向传播及参数更新等步骤。在每个epoch结束时，会在验证集上评估模型准确率。

6. **模型预测和pipeline**
   - 最后演示了如何利用训练好的模型对新句子进行情感分类预测，同时提供了批量化预测的方法。

## 05-Datasets的使用
   - 包含加载在线数据集 load_dataset
   - 加载数据集某项任务 load_dataset
   - 按照数据集划分进行加载 load_dataset
   - 查看数据集 index and slice
   - 数据集划分 train_test_split
   - 数据选取与过滤 select and filter
   - 数据映射 map
   - 保存与加载 save_to_disk / load_from_disk

   


---

此README文档提供了对Transformers库中模型加载、保存以及调用的基本指导。如果您是第一次接触这些内容，建议按照章节顺序逐步学习，以便更好地理解和掌握相关技能。