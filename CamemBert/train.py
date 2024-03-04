import os
import numpy as np
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import datasets
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
import sys

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import DatasetDict, Dataset
import logging
import transformers
import evaluate

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        help="Path to pretrained model or model identifier from huggingface.co/models",
                        required=True,
                        type=str)

    parser.add_argument("--data_folder_path",
                        help="The folder containing our training data",
                        required=True,
                        type=str)


    parser.add_argument("--output_dir",
                        help="The output directory to save trained model",
                        required=True,
                        type=str)

    parser.add_argument("--overwrite_output_dir",
                        help="When False and rerun the script, the training resume from the last checkpoint. If True, it restart training from scratch",
                        default=False)


    parser.add_argument("--hub_model_id",
                        help="The final model name to be push to hugging face for later inference",
                        required=True,
                        type=str)

    parser.add_argument("--text_column_name",
                        help="The name of the dataset column containing the text data. Defaults to 'phrase'",
                        default="phrase",
                        type=str)


    parser.add_argument("--target_column_name",
                        help="The name of the dataset column containing the target data.'",
                        default="catégorie",
                        type=str)

    parser.add_argument("--train_test_split",
                        help="Percentage threshold for training set",
                        default=0.2,
                        type=float)


    parser.add_argument("--do_train",
                        help="Specify whether or not to start training process",
                        default=True)

    parser.add_argument("--seed", default=42, type=int, help="random seed")

    parser.add_argument("--max_train_samples",
                        help="For debugging purposes or quicker training, truncate the number of training examples when this value is set",
                        default=None,
                        type=int)

    parser.add_argument("--max_eval_samples",
                        help="For debugging purposes or quicker training, truncate the number of eval examples to this value is set",
                        default=None,
                        type=int)


    parser.add_argument("--num_train_epochs",
                        default=3.,
                        type=float,
                        help = "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).")


    parser.add_argument("--per_device_train_batch_size",
                        default=16,
                        type=int,
                        help = "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.")
    

    parser.add_argument("--per_device_eval_batch_size",
                        default=16,
                        type=int,
                        help = "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.")
    

    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)

    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help = """If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.
                         For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until max_steps is reached
                        """)

    parser.add_argument("--evaluation_strategy",
                        default="steps",
                        type=str,
                        help = """The evaluation strategy to adopt during training. Possible values are:
                        "no": No evaluation is done during training.
                        "steps": Evaluation is done (and logged) every eval_steps.
                        "epoch": Evaluation is done at the end of each epoch.
                        """)

    parser.add_argument("--save_strategy",
                        default="steps",
                        type=str,
                        help = """The checkpoint save strategy to adopt during training. Possible values are:
                        "no": No saving is done during training.
                        "steps": saving is done (and logged) every eval_steps.
                        "epoch": saving is done at the end of each epoch.
                        """)

    parser.add_argument("--learning_rate",
                        default=2e-4,
                        type=float)

    parser.add_argument("--logging_steps",
                        default=10,
                        help = """Number of update steps between two logs if logging_strategy='steps'""",
                        type=int)

    parser.add_argument("--save_steps",
                        default=20,
                        help = """Number of updates steps before two checkpoint saves if save_strategy="steps".""",
                        type=int)

    parser.add_argument("--hub_strategy",
                        default="every_save",
                        help = """Defines the scope of what is pushed to the Hub and when""",
                        type=str)

    parser.add_argument("--push_to_hub",
                        help="Specify whether or not to save the model to hugging face for later inference",
                        default=True)

    return parser.parse_args()

# Utility functions

def read_data(data_path):
    """
    Reads data from a Excel file into a Pandas DataFrame.

    Params:
        data_path (str): The path to the Parquet file or folder containing the files to read.
    Returns:
        pandas.DataFrame: The DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(data_path)


def main():
  """
  Main function to train the classification model
  """
  args = parse_args()

  send_example_telemetry("run_classification_model", args)


  # Detecting last checkpoint.
  last_checkpoint = None
  if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
      raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
    elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

  # Set seed before initializing model.
  set_seed(args.seed)

  # 1. Create the dataset

  def map_target(target):
    if target == "recherche d'information":
      return 0
    elif target == 'Intention familière':
      return 1
    else:
      return 2

  logger.info(f"Buildind dataset")
  df = read_data(args.data_folder_path)
  df["label"] = df[args.target_column_name].apply(map_target)

  train, val = train_test_split(df, test_size=args.train_test_split, random_state=args.seed)

  raw_datasets = DatasetDict()

  raw_datasets['train'] = Dataset.from_pandas(train)
  raw_datasets['validation'] = Dataset.from_pandas(val)

  logger.info(f"Dataset built, switching to tokenization...")
  # Load model tokenizer

  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

  def preprocess_function(examples):
    """
    Preprocessing function to tokenize text and truncate sequences to be no longer than CamemBERT maximum input length:
    """
    return tokenizer(examples["phrase"], truncation=True)

  logger.info(f"Tokenizing dataset...")

  tokenized_dataset = raw_datasets.map(preprocess_function, batched=True, 
                                      remove_columns=[args.text_column_name, args.target_column_name])
  

  # Data Collator

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # Evaluate

  accuracy = evaluate.load("accuracy")


  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      return accuracy.compute(predictions=predictions, references=labels)


  id2label = {0: "recherche d'information", 1: 'Intention familière', 2: 'Action'}
  label2id = {v:k for k,v in id2label.items()}   


  # Load model
  model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                             num_labels=3, 
                                                             id2label=id2label, 
                                                             label2id=label2id)

  #initialize trainer
  trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    args=TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        evaluation_strategy = args.evaluation_strategy,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        hub_model_id = args.hub_model_id,
        push_to_hub = args.push_to_hub,
        save_strategy = args.save_strategy,
        save_steps = args.save_steps,
        hub_strategy = args.hub_strategy,
        overwrite_output_dir = args.overwrite_output_dir,


    ),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

  # Start training

  if args.do_train:
    logger.info(f"Starting the trainig process...")
  # use last checkpoint if exist
    if last_checkpoint is not None:
      checkpoint = last_checkpoint
    elif os.path.isdir(args.model_name_or_path):
      checkpoint = args.model_name_or_path
    else:
      checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (args.max_train_samples
            if args.max_train_samples is not None
            else len(tokenized_dataset["train"])
        )
    metrics["train_samples"] = min(max_train_samples, len(tokenized_dataset["train"]))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

  logger.info(f"Saving model")
  os.makedirs(args.output_dir, exist_ok = True)
  trainer.model.save_pretrained(args.output_dir)


  if args.push_to_hub:
    trainer.push_to_hub(args.hub_model_id)

if __name__ == "__main__":
    main()
