import os
import torch
import pandas as pd
import torch.distributed as dist

from tqdm import tqdm
from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer, BertForSequenceClassification


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    length = len(dataset)
    train_length = int(length * 0.9)  # 90%用于训练
    valid_length = length - train_length  # 剩余10%用于验证
    trainset, validset = random_split(dataset, lengths=[train_length, valid_length], generator=torch.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained("./chinese-roberta-wwm-ext-large")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=12, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained("./chinese-roberta-wwm-ext-large")

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()

    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerate: Accelerator, epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {ep+1}/{epoch}")
        for batch in pbar:
            with accelerate.accumulate(model):
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerate.backward(loss)
                optimizer.step()

                if accelerate.sync_gradients:
                    global_step += 1
                    if global_step % log_step == 0:
                        loss = accelerate.reduce(loss,"mean")
                        accelerate.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerate.log({'loss': loss.item()}, global_step)
                pbar.set_postfix({'loss': loss.item()})
        acc = evaluate(model, validloader, accelerate)
        accelerate.print(f"ep: {ep}, acc: {acc}")
        accelerate.log({'acc': acc}, global_step)
    accelerate.end_training()


def main():

    accelerator  = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=2, log_with='tensorboard', project_dir='ckpts')

    accelerator.init_trackers("runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":
    main()