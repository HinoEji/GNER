import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from argparse import ArgumentParser

from seqeval.metrics.sequence_labeling import get_entities
from evaluate import extract_predictions, parser
import json

def read_jsonl(path, tokenizer):
    data = []
  
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            token = obj['instance']['words']
            response = obj['prediction']
            bio_label = obj['instance']['labels']
            label_bounds = get_entities(bio_label)

            pred = {
                "label_list": obj['label_list'],
                "instance": {"words": token},
                "prediction": response,
            }
            bio_pred = extract_predictions(pred, tokenizer)
            entity_pred = parser(token, bio_pred)
            entity_label = parser(token, bio_label)

            data.append(
                {
                    "tokens": token,
                    "bio_labels" : bio_label,
                    "entity_labels": entity_label,
                    "bio_preds" : bio_pred,
                    "entity_preds" : entity_pred,
                    "response" : response,
                    "list_entities" : [x.lower() for x in obj['label_list']]
                }
            )


    return data


def main():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path to the prediction file")
    parser.add_argument("--output_name", type=str, default="formatted_predictions.jsonl", help="Name of the output file")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    samples = read_jsonl(args.prediction_path, tokenizer)
    with open(args.output_name, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

