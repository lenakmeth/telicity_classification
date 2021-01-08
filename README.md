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
* seaborn (not yet)

### Run 

`python classify_tensorflow.py --label_marker [telicity/duration] --model_type --[bert/roberta/xlnet/distilbert] --transformer_model [...] --num_epochs [2-4] --batch_size [...] --verg_segment_ids [yes/no]`

#### Arguments

* `label_marker`: Binary classification of telicity (telic/atelic) or duration (stative/dynamic). Default: _telicity_
* `model_type`: Supported model types (bert/roberta yes, xlnet/distilbert TBA). Default: _bert_
* `transformer_model`: The exact model. Default: _bert-base-uncased_
* `num_epochs`: Recommended 2-4. Default: _4_
* `batch_size`: Default: _32_
* `verb_segment_ids`: Whether we use token_type_ids or not to mark the verb position. Default: _no_. ATTN: RoBERTa and DistilBert do NOT support token_type_ids, always pass _no_. Also, the flag is NOT a boolean True/False, it's yes/no.
