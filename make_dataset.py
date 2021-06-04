from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pickle 

def remove_NA(d):
  """
  Remove null values and separate multilabel values with comma
  """
  if d['label'] == None:
    d['label'] = np.array('NA')
  if ' ' in d['label']:
    d['label'] = ",".join(sorted(d['label'].split()))
  return d

def label_encoding(d):
  """
  Split the multi-labels
  """
  d['label'] = np.array(str.upper(d['label']).split(","))
  return d


def make_dataset():


    dataset_0 = load_dataset(
        'csv', 
        data_files={
        'train': 'en_eacl/train.tsv', 
        'validation': 'en_eacl/dev.tsv', 
        'test': 'en_eacl/test.tsv'
        },
        delimiter='\t', 
        column_names=['label', 'sentence']
        )

    print("Removing null values:")
    dataset = dataset_0.map(remove_NA)
    print("Separating multilabels:")
    dataset = dataset.map(label_encoding)

    mlb = MultiLabelBinarizer()
  
    onehot_train = mlb.fit(dataset['train']['label'])
    labels = mlb.classes_
    print("Labels of the data:", labels)

    print("Binarizing the labels:")
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])})


    with open("en_binarized.pkl", 'wb') as f:
        pickle.dump(dataset,f)

    print(labels)



if __name__=="__main__":
    make_dataset()

