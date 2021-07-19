import argparse
import sys
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--training", default="yes", type=str, 
                        help="Do we train?")
    
    parser.add_argument("--label_marker", default="telicity", type=str, 
                        help="Whether the labels will be of telicity or duration.")
    
    parser.add_argument("--data_path", default="data/friedrich_captions_data", type=str, 
                        help="The datasets that will be used (acl_data = old / friedrich_captions_data = new / clean_data = cleaned friedrich but broken)")

    # transformer model
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, 
                        help="transformer model")

    parser.add_argument("--model_type", default="bert", type=str, 
                        help="type of the used transformer model: bert/roberta/xlnet/distilbert")

    # Number of training epochs (authors recommend between 2 and 4)
    parser.add_argument("--num_epochs", default=4, type=int, 
                        help="Number of training epochs (recommended 2-4).")

    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Recommended 16 or 32.")
    
    parser.add_argument("--verb_segment_ids", default="no", type=str, 
                        help="Whether we use the verb marking or not (only False for RoBERTa, DistilBert).")
    parser.add_argument("--freeze_layer_count", default=-1, type=int, 
                        help="")

    args = parser.parse_args()

    return args