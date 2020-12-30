import argparse
import sys
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()


    parser.add_argument("--type_prompts", default="nsubj", type=str, 
                        help="Options: nsubj/nsubj_amod/amod/dobj/dobj_amod")

    # transformer model
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, 
                        help="transformer model")
    # simpletransfromers needs the type of model, e.g. 'bert' for 'bert-base-uncased'
    # see full list https://huggingface.co/transformers/pretrained_models.html
    parser.add_argument("--model_type", default="bert", type=str, 
                        help="type of the used transformer model, see docs")

    # Number of training epochs (authors recommend between 2 and 4)
    parser.add_argument("--num_epochs", default=4, type=int, 
                        help="Number of training epochs (recommended 2-4).")

    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Recommended 16 or 32.")

    args = parser.parse_args()

    return args