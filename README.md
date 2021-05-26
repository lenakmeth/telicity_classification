# telicity_classification

## Experiment: classification with tensorflow

Code based on [aniruddhachoudhury](https://github.com/aniruddhachoudhury/BERT-Tutorials/tree/master/Blog%202). 

### Prerequisites

* pip > 19.0
* transformers
* torch >= 2.2.0
* keras
* sklearn

### Supported transformer models

#### English:
* bert: `bert-base|large-cased|uncased`
* roberta: `roberta-base|large`
* xlnet: `xlnet-base|large-cased`
* albert: `albert-base|large-v2`

#### French (careful with the spelling):
* FlauBERT: `flaubert/flaubert_small|base|large_cased` OR `flaubert/flaubert_base_uncased`
* CamemBERT: `camembert-base` OR `camembert/camembert-large`

### Run 

`python classify_tensorflow.py \
    --label_marker ['telicity' ,'duration'] \
    --path_data ['data/acl_data', 'data/clean_data', 'data/friedrich_captions_data']
    --transformer_model [...] \
    --num_epochs [2-4] \
    --batch_size [...] \
    --verb_segment_ids ['yes', 'no']
    --training ['yes', 'no']`

#### Arguments

* `label_marker`: Binary classification of telicity (telic/atelic) or duration (stative/dynamic). Default: _telicity_
* `path_data`: Which datasets will be used. 'acl' are the old Friedrich data, 'clean' are the cleaned but broken Friedrich data, 'friedrich_captions' (default) are the two datasets combined, 'french_data' are the French translations of the combined datasets.
* `transformer_model`: The exact model. Default: _bert-base-uncased_
* `num_epochs`: Recommended 2-4. Default: _4_
* `batch_size`: Default: _32_
* `verb_segment_ids`: Whether we use token_type_ids or not to mark the verb position. Default: _no_. ATTN: RoBERTa + Camembert models do NOT support token_type_ids, always pass _no_. Also, the flag is NOT a boolean True/False, it's yes/no.
* `training`: Whether we train the model or load a model from the checkpoints folder.
