# Lexical aspect binary classification (telicity/duration) with transformers

This is the code for the paper "About Time: Do Transformers Learn Temporal Lexical Aspect?" by E. Metheniti, T. Van de Cruys and N. Hathout. 

## Datasets

`data/en` and `data/fr` include a cleaned-up and concatenated version of the datasets by Friedrich and Gateva (2017) and Alikhani and Stone (2019) (English-French respectively)
`data/unseen_tests` includes our hand-crafted and annotated sentences for testing.

## Experiment: Finetuning and classification

### Prerequisites

* transformers
* torch >= 2.2.0
* sklearn

### Supported transformer models

#### English:
* BERT: `bert-{base,large}-{cased,uncased}`
* RoBERTa: `roberta-{base,large}`
* XLNet: `xlnet-{base,large}-cased`
* Albert: `albert-{base,large}-v2`

#### French (careful with the spelling):
* FlauBERT: `flaubert/flaubert_{small,base,large}_cased` OR `flaubert/flaubert_base_uncased`
* CamemBERT: `camembert-base` OR `camembert/camembert-large`

### Run 

`python finetune_classify.py \
    --label_marker {'telicity' ,'duration'} \
    --data_path {'data/en', 'data/fr'}
    --transformer_model {...} \
    --num_epochs {2-4} \
    --batch_size {...} \
    --verb_segment_ids {'yes', 'no'}
    --training {'yes', 'no'}`

#### Arguments

* `label_marker`: Binary classification of telicity (telic/atelic) or duration (stative/dynamic)
* `data_path`: Which datasets will be used. 
* `transformer_model`: The exact model. 
* `num_epochs`: Recommended 2-4. Default: _4_
* `batch_size`: Default: _32_
* `verb_segment_ids`: Whether we use token_type_ids or not to mark the verb position. Default: _no_. ATTN: RoBERTa + Camembert models do NOT support token_type_ids, always pass _no_. Also, the flag is NOT a boolean True/False, it's yes/no.
* `training`: Whether we train the model or load a model from the checkpoints folder.


## Experiment: Classification with pretrained embeddings & logistic regression

### Supported transformer models

#### English
See above.

### Run 

`python log_reg_embeddings.py \
    --label_marker {'telicity' ,'duration'} \
    --data_path {'data/en', 'data/fr'} \
    --transformer_model {...} \
    --batch_size {...} \
    --verb_segment_ids {'yes', 'no'}
    --freeze_layer_count {1-12/24}`

#### Arguments

See above, and also:
* `freeze_layer_count`: From which model layer we'll get the embeddings (max. 12 for base models, max. 24 for large models).

## Citation

@inproceedings{
metheniti2022about,
title={About Time: Do Transformers Learn Temporal Verbal Aspect?},
author={Metheniti, Eleni and Van de Cruys, Tim and Hathout, Nabil},
booktitle={Workshop on Cognitive Modeling and Computational Linguistics 2022},
year={2022},
url={https://openreview.net/forum?id=Bduz_QYg8Z5}
}