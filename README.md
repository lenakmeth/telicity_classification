# telicity_classification

## Experiment one: classification w/ `simpletransformers`

### Prerequisites

* [simpletransformers](https://pypi.org/project/simpletransformers/): `pip install simpletransformers`
* torch
* pandas

### Run 

`python classify_simpletransformers.py [optional:directory of sentences]`

Edit model arguments on line 53.

### Results 

* 5 epochs, bert-base-uncased
    - eval_loss = 2.0004521620731732
    - fn = 203
    - fp = 240
    - mcc = 0.30941834056538914
    - tn = 415
    - tp = 423


## Experiment two: classification with tensorflow

Code based on [aniruddhachoudhury](https://github.com/aniruddhachoudhury/BERT-Tutorials/tree/master/Blog%202). 

### Prerequisites

* pip > 19.0
* transformers
* torch >= 2.2.0
* keras
* sklearn
* seaborn

### Run 

`python classify_tensorflow.py`

Check arguments on lines 19-26 (no config file yet, sorry!).

### Results

* 4 epochs, bert-base-uncased
    - Average training loss: 0.38
    - Accuracy: 0.69