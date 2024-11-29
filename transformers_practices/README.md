
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
    

