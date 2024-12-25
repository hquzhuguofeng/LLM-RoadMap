from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import evaluate

# Step1: 加载数据集
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)

# Step2: 划分数据集
datasets = dataset.train_test_split(test_size=0.1, seed=42)

# Step3: 数据集预处理
tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=["review", "label"])

# Step4: 创建模型
model = BertForSequenceClassification.from_pretrained("/gemini/code/model")

# Step5: 创建评估函数
acc_metric = evaluate.load("./metric_accuracy.py")
f1_metric = evaluate.load("./metric_f1.py")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {**acc, **f1}

# Step6: 创建TrainingArguments
train_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True
)

# Step7: 创建Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)

# Step8: 模型训练
trainer.train()