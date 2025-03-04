
import os
import json

file_path = './data/train/train_data_自我认知.jsonl'
output_path = './data/train/train_data_自我认知_plus.jsonl'
name = '小安'
company = '心灵树洞'
years = '2025'


res = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        answer = entry.get('answer')
        answer = answer.replace('[名字代号]', name).replace('[公司代号]', company).replace('[时间代号]', years)

        i, question = entry.get('index'), entry.get('question')
        output = {'index': i, 'question': question, 'answer': answer}
        res.append(output)


def write_output(output_filename, data):
    with open(output_filename, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

write_output(output_path, res)
print("all done")


        