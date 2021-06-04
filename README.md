# multilabel_explainability
Code for training a multilabel register model and explaining its predictions. Based heavily on my register_explainability for multiclass classification, which in turn is based on TurkuNLP/explain-bert by Filip Ginter.

1. Run make_dataset.py to binarize the labels. Remember to copy the labels that are printed as an array!
2. Train the model with train_multilabel.py. Paste the labels to marked line.
3. explain_multilabel.py saves the aggregation scores of each word/label pair to a tsv-file. Aggregation score describes the importance of the word when the model makes the classification.
4. keywords_multilabel.py saves the most important ones of these words.
