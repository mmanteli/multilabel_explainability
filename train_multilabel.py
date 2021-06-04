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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Hyperparameters
LEARNING_RATE=3e-5
BATCH_SIZE=4
TRAIN_EPOCHS=6
MODEL_NAME = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# these are printed out by make_dataset.py 
labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']

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
  
  with open('en_binarized.pkl', 'rb') as f:
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
    # flatten them to 1D vectors
    y_pred = pred.predictions.flatten()
    y_true = pred.label_ids.flatten()

    # apply sigmoid
    y_pred_s = 1.0/(1.0 + np.exp(-y_pred))

    threshold = 0.5  
    pred_ones = [pl>threshold for pl in y_pred_s]
    true_ones = [tl==1 for tl in y_true]
    return { 'accuracy': accuracy_score(pred_ones, true_ones) }



def train(dataset):
  
  # Model downloading
  num_labels = len(dataset['train']['label'][0][0]) #here double brackets are needed!
  print("Downloading model")
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = num_labels)
  
  print("Tokenizing data")
  encoded_dataset = dataset.map(encode_dataset)
  
  train_args = TrainingArguments(
        'multilabel_model_checkpoints',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
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
  
  print('Accuracy:', results["eval_accuracy"])
  
  # for classification report
  val_predictions = trainer.predict(encoded_dataset['validation'])

  # sanity check; did we even predict anything (a real consern)
  #counter = 0
  #predic_s = 1.0/(1.0+ np.exp(- val_predictions.predictions.flatten()))
  #for i in range(len(predic_s)):
    #if predic_s[i] > 0.5:
      #counter += 1
  #print("predicted ones in all predictions (sanity check): ", counter)
  
  # apply sigmoid to predictions and reshape real labels
  p = 1.0/(1.0 + np.exp(- val_predictions.predictions))
  t = val_predictions.label_ids.reshape(p.shape)

  pred_ones = [pl>0.5 for pl in p]
  true_ones = [tl==1 for tl in t]

  # labels are printed by make_dataset.py copy them there to hyperparametres
  print(classification_report(true_ones,pred_ones, target_names = labels))

  # save the model
  torch.save(trainer.model,"multilabel_model_en_2.pt")

if __name__=="__main__":
  
  device = 'cuda' if cuda.is_available() else 'cpu'
  dataset = read_dataset()
  train(dataset)
