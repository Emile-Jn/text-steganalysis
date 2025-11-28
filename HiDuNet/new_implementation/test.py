
# Third-party modules
import logging
import torch
from transformers import BertTokenizer

import HiDuNet
# Project modules
from HiDuNet.models.configuration_elasticbert import ElasticBertConfig
from HiDuNet.models.modeling_elasticbert_entropy import ElasticBertForSequenceClassification
from HiDuNet.arguments import get_args
from data_loader import TSVTextDataset
from utils import get_root_dir, get_device


def example_inference(args=None):
    """Run inference in the simplest way possible."""
    # if args is None:
    #    args = get_args()

    model_name_or_path = "bert-base-uncased"

    # Load pretrained config and tokenizer from model_name_or_path
    config = ElasticBertConfig.from_pretrained(
        model_name_or_path
    )

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    embed_num = tokenizer.vocab_size

    # Load the model directly from model_name_or_path
    model = ElasticBertForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        args=None,
        add_pooling_layer=True,
    )

    model.to(get_device())

    # Load data
    data = TSVTextDataset(
        get_root_dir() / "data" / "test_inference" / "test.tsv",
        tokenizer
    )

    # Run inference
    preds = [] # predicted labels
    model.eval()
    for batch in data:
        with torch.no_grad():
            inputs = {
                # "input_ids": batch[0].unsqueeze(0),
                # "attention_mask": batch[1].unsqueeze(0),
                # "token_type_ids": batch[2].unsqueeze(0),
                "input_ids": batch["input_ids"].unsqueeze(0),
                "attention_mask": batch["attention_mask"].unsqueeze(0)
            }
            outputs = model(**inputs)
            logits = outputs[0]
            predicted_class = torch.argmax(logits, dim=1).item()
            preds.append(predicted_class)
            print(f"Predicted class: {predicted_class}")
    return preds

if __name__ == "__main__":
    example_inference()