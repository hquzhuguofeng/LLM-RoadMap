
import logging
import sys

from dataclasses import dataclass
from utils.model import Mymodel
from utils.data import TrainDatasetForOrder,GroupCollator
from utils.trainer import MyTrainer
from transformers import TrainingArguments, HfArgumentParser

logger = logging.getLogger(__name__)

@dataclass
class DataAndModelArguments:
    dataset_dir : str
    cache_dir : str

def main():
    parser = HfArgumentParser((TrainingArguments, DataAndModelArguments))
    training_args, data_model_args = parser.parse_args_into_dataclasses()
    training_args : TrainingArguments
    data_model_args : DataAndModelArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Training parameters {training_args}")

    model = Mymodel()
    dataset = TrainDatasetForOrder(data_model_args.dataset_dir, data_model_args.cache_dir)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=GroupCollator()
    )
    trainer.train()

if __name__ == "__main__":
    main()