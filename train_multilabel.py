from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import datasets
import pickle
import sys
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
import numpy as np


# Hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
TRAIN_EPOCHS=6
MODEL_NAME = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# overriding the loss-function in Trainer
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def read_dataset():
  """
  Read the data. Labels should be in the form of binary vectors.
  """
  
  with open('fr_binarized.pkl', 'rb') as f:
  dataset = pickle.load(f)
  print("Dataset succesfully loaded")
  print(dataset)
  return dataset


def encode_dataset(d):
  """
  Tokenize the sentences. Null/None sentences converted to empty strings.
  """
  try:
    output = tokenizer(d['sentence'], truncation= True, padding = True, max_length=512)
    return output
  except:     #there were a few empty sentences
    output = tokenizer(" ", truncation= True, padding = True, max_length=512)
    return output

def compute_accuracy(pred):
  y_pred = pred.predictions.argmax(axis=1)
  y_true = pred.label_ids
  return { 'accuracy': sum(y_pred == y_true) / len(y_true) }




def train(dataset):
  
  # Model downloading
  num_labels = len(dataset['train']['label'][0][0])
  print("Downloading model")
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = num_labels)
  
  print("Tokenizing data")
  encoded_dataset = dataset.map(encode_dataset)
  
  train_args = TrainingArguments(
        'multilabel_model_checkpoints',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        gradient_accumulation_steps=4,
        save_total_limit=3
    )
  
  
  trainer = MultilabelTrainer(
        model,
        train_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )
  
  print("Ready to train")
  trainer.train()
  
  results = trainer.evaluate()
  print(f'Accuracy: {results["eval_accuracy"]}')
  
  


if __name__=="__main__":
  
  device = 'cuda' if cuda.is_available() else 'cpu'
  dataset = read_dataset()
  train(dataset)
