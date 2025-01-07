## 文本匹配任务（Text Matching）



本项目对4种常用的文本匹配的方法进行实现：PointWise（单塔）、DSSM（双塔）、Sentence BERT（双塔）、SIMCSE、ESIMCSE。



## 1. 环境安装

本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
torch
transformers==4.22.1
datasets==2.4.0
evaluate==0.2.2
matplotlib==3.6.0
rich==12.5.1
scikit-learn==1.1.2
requests==2.28.1
```

## 2. 数据集准备

项目中提供了一部分示例数据，我们使用「商品评论」和「商品类别」来进行文本匹配任务，数据在 `data/comment_classify` 。

若想使用`自定义数据`训练，只需要仿照示例数据构建数据集即可：

```python
衣服：指穿在身上遮体御寒并起美化作用的物品。	为什么是开过的洗发水都流出来了、是用过的吗？是这样子包装的吗？	0
衣服：指穿在身上遮体御寒并起美化作用的物品。	开始买回来大很多 后来换了回来又小了 号码区别太不正规 建议各位谨慎	1
...
```

每一行用 `\t` 分隔符分开，第一部分部分为`商品类型（text1）`，中间部分为`商品评论（text2）`，最后一部分为`商品评论和商品类型是否一致（label）`。


## 3. 模型训练

### 3.1 PointWise（单塔） 

#### 3.1.1 模型训练

修改训练脚本 `train_pointwise.sh` 里的对应参数, 开启模型训练：

```sh
python train_pointwise.py \
    --model "nghuyong/ernie-3.0-base-zh" \  # backbone
    --train_path "data/comment_classify/train.txt" \    # 训练集
    --dev_path "data/comment_classify/dev.txt" \    #验证集
    --save_dir "checkpoints/comment_classify" \ # 训练模型存放地址
    --img_log_dir "logs/comment_classify" \ # loss曲线图保存位置
    --img_log_name "ERNIE-PointWise" \  # loss曲线图保存文件夹
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
global step 10, epoch: 1, loss: 0.77517, speed: 3.43 step/s
...
global step 100, epoch: 3, loss: 0.31319, speed: 4.01 step/s
Evaluation precision: 0.96970, recall: 0.90566, F1: 0.93659
best F1 performence has been updated: 0.88152 --> 0.93659
...
```


#### 3.1.2 模型推理

完成模型训练后，运行 `inference_pointwise.py` 以加载训练好的模型并应用：

```python
...
    test_inference(
        '手机：一种可以在较广范围内使用的便携式电话终端。',     # 第一句话
        '味道非常好，京东送货速度也非常快，特别满意。',        # 第二句话
        max_seq_len=128
    )
...
```

运行推理程序：

```sh
python inference_pointwise.py
```

得到以下推理结果：

```sh
tensor([[ 1.8477, -2.0484]], device='cuda:0')   # 两句话不相似(0)的概率更大
```

---

### 3.2 DSSM（双塔）

#### 3.2.1 模型训练

修改训练脚本 `train_dssm.sh` 里的对应参数, 开启模型训练：

```sh
python train_dssm.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/dssm" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "ERNIE-DSSM" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
global step 0, epoch: 1, loss: 0.62319, speed: 15.16 step/s
Evaluation precision: 0.29912, recall: 0.96226, F1: 0.45638
best F1 performence has been updated: 0.00000 --> 0.45638
...
global step 50, epoch: 1, loss: 0.30996, speed: 3.68 step/s
...
```


#### 3.2.2 模型推理

和单塔模型不一样的是，双塔模型可以事先计算所有候选类别的Embedding，当新来一个句子时，只需计算新句子的Embedding，并通过余弦相似度找到最优解即可。

因此，在推理之前，我们需要提前计算所有类别的Embedding并保存。

> 类别Embedding计算

运行 `get_embedding.py` 文件以计算对应类别embedding并存放到本地：

```python
...
text_file = 'data/comment_classify/types_desc.txt'                       # 候选文本存放地址
output_file = 'embeddings/comment_classify/dssm_type_embeddings.json'    # embedding存放地址

device = 'cuda:0'                                                        # 指定GPU设备
model_type = 'dssm'                                                      # 使用DSSM还是Sentence Transformer
saved_model_path = './checkpoints/comment_classify/dssm/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
...
```

其中，所有需要预先计算的内容都存放在 `types_desc.txt` 文件中。

文件用 `\t` 分隔，分别代表 `类别id`、`类别名称`、`类别描述`：

```txt
0	水果	指多汁且主要味觉为甜味和酸味，可食用的植物果实。
1	洗浴	洗浴用品。
2	平板	也叫便携式电脑，是一种小型、方便携带的个人电脑，以触摸屏作为基本的输入设备。
...
```

执行 `python get_embeddings.py` 命令后，会在代码中设置的embedding存放地址中找到对应的embedding文件：

```json
{
    "0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [0.3363891839981079, -0.8757723569869995, -0.4140555262565613, 0.8288457989692688, -0.8255823850631714, 0.9906797409057617, -0.9985526204109192, 0.9907819032669067, -0.9326567649841309, -0.9372553825378418, 0.11966298520565033, -0.7452883720397949,...]},
    "1": ...,
    ...
}
```

> 模型推理

完成预计算后，接下来就可以开始推理了。

我们构建一条新评论：`这个破笔记本卡的不要不要的，差评`。

运行 `python inference_dssm.py`，得到下面结果：

```python
[
    ('平板', 0.9515482187271118),
    ('电脑', 0.8216977119445801),
    ('洗浴', 0.12220608443021774),
    ('衣服', 0.1199738010764122),
    ('手机', 0.07764233648777008),
    ('酒店', 0.044791921973228455),
    ('水果', -0.050112202763557434),
    ('电器', -0.07554933428764343),
    ('书籍', -0.08481660485267639),
    ('蒙牛', -0.16164332628250122)
]
```
函数将输出（类别，余弦相似度）的二元组，并按照相似度做倒排（相似度取值范围：[-1, 1]）。

---

### 3.3 Sentence Transformer（双塔）

#### 3.3.1 模型训练

修改训练脚本 `train_sentence_transformer.sh` 里的对应参数, 开启模型训练：

```sh
python train_sentence_transformer.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/sentence_transformer" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "Sentence-Ernie" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
Evaluation precision: 0.81928, recall: 0.64151, F1: 0.71958
best F1 performence has been updated: 0.46120 --> 0.71958
...
global step 300, epoch: 2, loss: 0.56033, speed: 3.55 step/s
...
```


#### 3.3.2 模型推理

Sentence Transformer 同样也是双塔模型，因此我们需要事先计算所有候选文本的embedding值。

> 类别Embedding计算

运行 `get_embedding.py` 文件以计算对应类别embedding并存放到本地：

```python
...
text_file = 'data/comment_classify/types_desc.txt'                       # 候选文本存放地址
output_file = 'embeddings/comment_classify/sentence_transformer_type_embeddings.json'    # embedding存放地址

device = 'cuda:0'                                                        # 指定GPU设备
model_type = 'sentence_transformer'                                                      # 使用DSSM还是Sentence Transformer
saved_model_path = './checkpoints/comment_classify/sentence_transformer/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
...
```

其中，所有需要预先计算的内容都存放在 `types_desc.txt` 文件中。

文件用 `\t` 分隔，分别代表 `类别id`、`类别名称`、`类别描述`：

```txt
0	水果	指多汁且主要味觉为甜味和酸味，可食用的植物果实。
1	洗浴	洗浴用品。
2	平板	也叫便携式电脑，是一种小型、方便携带的个人电脑，以触摸屏作为基本的输入设备。
...
```

执行 `python get_embeddings.py` 命令后，会在代码中设置的embedding存放地址中找到对应的embedding文件：

```json
{
    "0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [0.32447007298469543, -1.0908259153366089, -0.14340722560882568, 0.058471400290727615, -0.33798110485076904, -0.050156619399785995, 0.041511114686727524, 0.671889066696167, 0.2313404232263565, 1.3200652599334717, -1.10829496383667, 0.4710233509540558, -0.08577515184879303, -0.41730815172195435, -0.1956728845834732, 0.05548520386219025, ...]}
    "1": ...,
    ...
}
```

> 模型推理

完成预计算后，接下来就可以开始推理了。

我们构建一条新评论：`这个破笔记本卡的不要不要的，差评`。

运行 `python inference_sentence_transformer.py`，函数会输出所有类别里「匹配通过」的类别及其匹配值，得到下面结果：

```python
Used 0.5233056545257568s.
[
    ('平板', 1.136274814605713), 
    ('电脑', 0.8851938247680664)
]
```
函数将输出（匹配通过的类别，匹配值）的二元组，并按照匹配值（越大则越匹配）做倒排。


---

### 3.4 SimCSE: 无监督文本匹配模型

文本匹配多用于计算两个文本之间的相似度，该示例会基于 ESimCSE 实现一个无监督的文本匹配模型的训练流程。核心原理：由于预训练模型在训练的时候通常都会dropout，意味着即使是同一个样本过两次模型也会得到不同的embedding。而因为同样的样本，那一定是相似的，模型输出的这两个embedding局里就应当尽可能的相近；反之，那些不同的输入样本过模型得到的embedding尽可能的被退远。

更详细解释[超细节的对比学习和SimCSE知识点](https://zhuanlan.zhihu.com/p/378340148)

simcse的缺点：所有的正例都是由「同一个句子」过了两次模型得到的。这就会造成一个问题：模型会更倾向于认为，长度相同的句子就代表一样的意思。由于数据样本是随机选取的，那么很有可能在一个 batch 内采样到的句子长度是不相同的。解决办法：[ESimCSE](https://arxiv.org/pdf/2109.04380), 通过duplicate_ratio超参数控制。

#### 3.4.1 数据准备

项目中提供了一部分示例数据，我们使用未标注的用户搜索记录数据来训练一个文本匹配模型，数据在 `data/LCQMC` 。

若想使用`自定义数据`训练，只需要仿照示例数据构建数据集即可：

* 训练集：

```python
喜欢打篮球的男生喜欢什么样的女生
我手机丢了，我想换个手机
大家觉得她好看吗
求秋色之空漫画全集
晚上睡觉带着耳机听音乐有什么害处吗？
学日语软件手机上的
...
```

* 测试集：

```python
开初婚未育证明怎么弄？	初婚未育情况证明怎么开？	1
谁知道她是网络美女吗？	爱情这杯酒谁喝都会醉是什么歌	0
人和畜生的区别是什么？	人与畜生的区别是什么！	1
男孩喝女孩的尿的故事	怎样才知道是生男孩还是女孩	0
...
```
由于是无监督训练，因此训练集（train.txt）中不需要记录标签，只需要大量的文本即可。

测试集（dev.tsv）用于测试无监督模型的效果，因此需要包含真实标签。

每一行用 `\t` 分隔符分开，第一部分部分为`句子A`，中间部分为`句子B`，最后一部分为`两个句子是否相似（label）`。

#### 3.4.2 模型训练

修改训练脚本 `train.sh` 里的对应参数, 开启模型训练：

```sh
python train_simcse.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/LCQMC/train.txt" \
    --dev_path "data/LCQMC/dev.tsv" \
    --save_dir "checkpoints/LCQMC" \
    --img_log_dir "logs/LCQMC" \
    --img_log_name "ERNIE-ESimCSE" \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --batch_size 64 \
    --max_seq_len 64 \
    --valid_steps 400 \
    --logging_steps 50 \
    --num_train_epochs 8 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 226.41it/s]
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 477532
    })
    dev: Dataset({
        features: ['text'],
        num_rows: 8802
    })
})
global step 50, epoch: 1, loss: 0.34367, speed: 2.01 step/s
...
global step 350, epoch: 1, loss: 0.06673, speed: 2.01 step/s
global step 400, epoch: 1, loss: 0.05954, speed: 1.99 step/s
Evaluation precision: 0.58459, recall: 0.87210, F1: 0.69997, spearman_corr: 
0.36698
best F1 performence has been updated: 0.00000 --> 0.69997
global step 450, epoch: 1, loss: 0.25825, speed: 2.01 step/s
...
global step 650, epoch: 1, loss: 0.26931, speed: 2.00 step/s
...
```


#### 3.4.3 模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```python
...
    if __name__ == '__main__':
    ...
    sentence_pair = [
        ('男孩喝女孩的故事', '怎样才知道是生男孩还是女孩'),
        ('这种图片是用什么软件制作的？', '这种图片制作是用什么软件呢？')
    ]
    ...
    res = inference(query_list, doc_list, model, tokenizer, device)
    print(res)
```

运行推理程序：

```sh
python inference.py
```

得到以下推理结果：

```python
[0.1527191698551178, 0.9263839721679688]   # 第一对文本相似分数较低，第二对文本相似分数较高
```

#### 3.4.4 模型推理加速
使用onnxruntime加速sbert的推理速度
详情见 [07-04-sbert_onnx.ipynb](./07-04-sbert_onnx.ipynb)

