

# 领域self-instruct

> 按照self-instruct的逻辑，基于GPT3.5的接口实现垂直领域数据集的自动构建，并不区分分类和生成的答案类型
>
> 其中的重点在于根据自己的领域问题来编写prompt，在config.py配置中。
>
> 当前项目已经生成500+数据，如果需要生成更多数据修改`config.py num_total_generate`


# 一、项目结构

```bash
├── ./config.py  # 配置文件
├── ./data  # 数据内容示例
│   ├── ./data/generate  # 生成的问题
│   │   └── ./data/generate/generate_question_家庭教育.jsonl  
│   ├── ./data/seed  # 人工或者提前定义好的种子问题
│   │   └── ./data/seed/seed_question_家庭教育.jsonl
│   └── ./data/train  # 最后生成的问答对
│       ├── ./data/train/train_data_家庭教育.jsonl
│       └── ./data/train/train_data_自我认知.jsonl  # 自我认知这种数据需要人工编辑问题和答案
│       └── ./data/train/train_data_自我认知_plus.jsonl  # 修改后的数据
├── ./domain_self_answer.py  # 领域回答生成脚本
├── ./domain_self_question.py  # 领域问题生成脚本
└── ./README.md
```

# 三、使用

## 1.已经基于OpenAI接口实现q2r函数，如果需要支持其他模型，则自定义修改函数

> step1 `gpt-3.5-turbo`的接口和密钥，参考链接[DevAGI手册](https://docs.devcto.com/quickstart)

> step2 在网页中[DevAGI 我的Keys](https://devcto.com/app/account)中找到自己的密钥

> step3 在`.env`中填写自己的`API URL`:`https://api.fe8.cn/v1` 和 密钥



## 2.编辑种子问题，并配置生成问题的参数并运行 [domain_self_question.py](domain_self_question.py) 

> 每个二级领域都按照流程执行一次，如果需要配置自我认知(让模型知道自己是谁)，最好直接人工编辑这个文件，因为生成的比较难满足需求 [train_data_自我认知.jsonl](data/train/train_data_自我认知.jsonl) 

> python self_regonition.py

### a.编辑种子问题

示例： [seed_question_家庭教育.jsonl](data/seed/seed_question_家庭教育.jsonl) 

```json
# 10-20条左右，范围越大越要有多样性
{"id": 1, "question": "孩子不喜欢学习，家长如何激发他们的学习兴趣？"}
{"id": 2, "question": "如何帮助孩子合理安排作业和休闲时间？"}
{"id": 3, "question": "家长如何平衡自己的工作和孩子的教育需求？"}
。。。
```

### b.配置生成问题参数

示例： [config.py](config.py) 

```python
# domain_self_question.py 配置内容
domain = '家庭教育'  # 二级领域之一
seed_tasks_file = "./data/seed/seed_tasks_%s.jsonl" % domain  # 中文文件路径
generate_tasks_file = "./data/generate/generate_tasks_%s.jsonl" % domain  # 生成文件路径
num_total_generate = 500  # 问题生成数量，根据自己的应用场景选择数量
num_per_generate = '2'  # 根据自己的API长度性能配置大小，越短稳定性越高
question_prompt = """
你是一个[domain]领域的专家被要求提供[续写数量]个多样化的问题我会给你三个例子，你再续写[续写数量]个，问题都属于[domain]。
以下是你提供指令需要满足的要求：
1.尽量不要在每个指令中重复动词，要最大化指令的多样性，但是内容都属于[domain]
2.使用指令的语气要符合中国的家长和孩子。
下面是[例子数量]个例子:
[例子生成]
下面请续写[续写数量]个问题，格式保持跟上面类似的序号,用续写1.续写2.这样以此类推的格式,不要用其他多余的符号就是续写+数字+.:
"""  # 问题生成prompt
```

### c.运行domain_self_question.py

```python
python domain_self_question.py

"""输出示例(已经生成过了，没生过会显示生成的信息)
所需的生成数量: 0
"""
```

## 3.配置生成答案参数并运行 [domain_self_answer.py](domain_self_answer.py) 

### a.配置生成答案参数

示例： [config.py](config.py) 

```python
# domain_self_answer.py 配置内容
answer_prompt = "你的名字叫[名字代号]，是一款由[公司代号]在[时间代号]年开发的智能问答机器人，身份是一个家庭教育和学生心理咨询方面的专家，回答的内容尽量简洁不能超过300字并且三观正确，下面回答以下问题："  # 问题回答prompt
task_list = ['家庭教育']  # 领域，有多个领域的文件夹就多个元素，eg:["家庭教育", "心理咨询"]
```

### b.运行domain_self_answer.py

```python
python domain_self_answer.py

"""输出示例(已经生成过了，没生过会显示生成的信息)
console:
本次任务类别: 家庭教育
本次任务问题数量： 500
第0个
。。。
问题已存在
第498个
问题已存在
第499个
问题已存在
500it [00:00, 439286.13it/s]
"""
```

## 4.参考
- [self-instruct github](https://github.com/yizhongw/self-instruct)
- [self-instruct paper](https://arxiv.org/abs/2212.10560)
- [alpaca self instruct](https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py)