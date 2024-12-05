import collections
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, pipeline
from cmrc_eval import evaluate_cmrc

def load_data():
    # 加载数据集
    datasets = load_dataset('cmrc2018', cache_dir='./data')
    print(datasets['train'][0])
    return datasets

def preprocess_data(datasets):
    tokenizer = AutoTokenizer.from_pretrained('D:/pretrained_model/models--hfl--chinese-macbert-base')
    
    return tokenizer

def process_function(examples, tokenizer):
    tokenizer_examples = tokenizer(
        text=examples['question'],
        text_pair=examples['context'],
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        stride=128,
        max_length=512,
        truncation='only_second',
        padding='max_length'
    )
    sample_mapping = tokenizer_examples.pop('overflow_to_sample_mapping')

    start_position = []
    end_position = []
    examples_ids = []  # 用于记录答案是在原来段落中的哪个段落的
    for idx, _ in enumerate(sample_mapping):
        answer = examples["answers"][sample_mapping[idx]]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        context_start = tokenizer_examples.sequence_ids(idx).index(1)
        context_end = tokenizer_examples.sequence_ids(idx).index(None, context_start) - 1
        offset = tokenizer_examples.get("offset_mapping")[idx]

        if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id

            token_id = context_end
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id

        start_position.append(start_token_pos)
        end_position.append(end_token_pos)
        examples_ids.append(examples['id'][sample_mapping[idx]])

        tokenizer_examples['offset_mapping'][idx] = [
            (o if tokenizer_examples.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(tokenizer_examples['offset_mapping'][idx])
        ]
    
    tokenizer_examples["example_ids"] = examples_ids
    tokenizer_examples['start_positions'] = start_position
    tokenizer_examples['end_positions'] = end_position
    return tokenizer_examples

def get_result(start_logits, end_logits, examples, features):
    predictions = {}
    references = {}

    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features["example_ids"]):
        example_to_feature[example_id].append(idx)

    n_best = 20
    max_answer_length = 30

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_idx in example_to_feature[example_id]:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offset = features[feature_idx]["offset_mapping"] # 分词后的偏移
            start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    answers.append({
                        "text": context[offset[start_index][0]: offset[end_index][1]],
                        "score": start_logit[start_index] + end_logit[end_index]
                    })
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
        references[example_id] = example["answers"]["text"]

    return predictions, references

def metric(pred,):
    start_logits, end_logits = pred[0]
    if start_logits.shape[0] == len(tokenizer_datasets["validation"]):
        p, r = get_result(start_logits, end_logits, datasets["validation"], tokenizer_datasets["validation"])
    else:
        p, r = get_result(start_logits, end_logits, datasets["test"], tokenizer_datasets["test"])
    return evaluate_cmrc(p, r)
    

if __name__ == '__main__':
    # Step 1: Load dataset
    datasets = load_data()

    # Step 2: Preprocess data
    tokenizer = preprocess_data(datasets)

    # Tokenize entire dataset
    tokenizer_datasets = datasets.map(
        function=lambda examples: process_function(examples, tokenizer),
        batched=True,
        remove_columns=datasets['train'].column_names
    )
    
    # Step 4: Create model
    model = AutoModelForQuestionAnswering.from_pretrained('D:/pretrained_model/models--hfl--chinese-macbert-base')

    args = TrainingArguments(
        output_dir='./models_for_qa',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        max_steps=80
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenizer_datasets['train'],
        eval_dataset=tokenizer_datasets['validation'],
        data_collator=DefaultDataCollator(),
        compute_metrics=metric
    )

    # Step 5: Train model
    trainer.train()

    # Step 9: Inference
    pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
    question = "小明在哪里上班？"
    context = "小明在上海工作过，现在在深圳做了。"
    result = pipe(question=question, context=context)
    print(result)
