import logging
import os
import torch
import transformers

from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path : Optional[str] = field(default="/home/szem/szsti_project/guofeng/pretrained_model/Qwen/Qwen2___5-3B-Instruct")
    use_lora : Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path : str = field(
        default=None, metadata={"help" : "Path to the training data."}
    )
    source_length : int = field(default=128)
    target_length : int = field(default=512)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir : Optional[str] = field(default=None)
    optim : str = field(default="adamw_torch")
    model_max_length : int = field(
        default=512,
        metadata={"help": "Maximum sequence length, sequeces will be right padded"}
    )
    use_deepspeed : bool = field(default=False)

def get_all_datapath(dir_name : str) -> List[str]:
    all_file_list = []

    for root, dir, file_names in os.walk(dir_name):
        for item in file_names:
            standard_file_path = f"{root}/{item}"
            all_file_list.append(standard_file_path)
    return all_file_list

def load_dataset_from_path(
        data_path : Optional[str] = None, cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split('.')[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir
    )['train']
    return raw_datasets

def find_subsequence(main_seq: List[int], sub_seq: List[int]) -> List[int]:
    position = []
    for i in range(len(main_seq) - len(sub_seq) + 1):
        if main_seq[i : i + len(sub_seq)] == sub_seq:
            position.append(i)
    return position

def load_train_dataset(
    tokenizer=PreTrainedTokenizer,
    data_path=str,
    data_args=DataArguments
):
    # dataset = None
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(data_path)

    logging.warning("Formatting inputs...")

    def generate_soures_targets(
        examples: Dict, tokenizer: PreTrainedTokenizer
    ):
        instruction = examples['instruction']
        output = examples['output']

        instruction_length = len(instruction)

        '''
        <|im_start|>system
        你是一个非常厉害的医生，精通各种医术，现在有患者开始向你描述他的情况，请帮帮他<|im_end|>
        <|im_start|>user
        患者描述的内容：


        我有头疼<|im_end|>
        <|im_start|>assistant
        去看医生<|im_end|>
        '''
        def build_message2ids(ins_data, output):
            prompt = f"患者描述的内容：\n\n\n {ins_data}"
            message = [
                {
                    "role":"system",
                    "content":"你是一个非常厉害的医生，精通各种医术，现在有患者开始向你描述他的情况，请帮帮他"
                },
                {
                    "role":"user",
                    "content":prompt
                },
                {
                    "role":"assistant",
                    "content":output
                }
            ]
            token_id_list = tokenizer.apply_chat_template(message)
            sub_sequence = [77091]
            last_gen_id = find_subsequence(token_id_list, sub_sequence)[-1] + 2

            input_ids = token_id_list.copy()

            labels = [-100] * last_gen_id + token_id_list[
                (last_gen_id - len(token_id_list)) :
            ]

            return input_ids, labels


        input_ids = []
        labels = []

        for i in range(instruction_length):
            temp_input_ids, temp_labels = build_message2ids(instruction[i], output[i])
            input_ids.append(temp_input_ids)
            labels.append(temp_labels)
        
        examples["input_ids"] = input_ids
        examples["labels"] = labels
        return examples
    
    generate_soures_targets_p = partial(generate_soures_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_soures_targets_p,
        batched=True,
        desc="running tokenizer on train dataset",
        num_proc=4
    ).shuffle()

    return dataset

def load_model_and_tokenizer(
    model_args : ModelArguments,
    training_args : TrainingArguments
):
    if training_args.use_deepspeed:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True
        ) # deepspeed会自动接管
    else:
        # 需要自己指定
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        ) 
    
    if model_args.use_lora:
        logging.warning("loading model by lora")

        from peft import get_peft_model, LoraConfig, TaskType

        LORA_R = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj']

        lora_config = LoraConfig(
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    return peft_model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_and_tokenizer(model_args=model_args, training_args=training_args)

    with training_args.main_process_first(desc="loading and process data"):
        train_dataset = load_train_dataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()