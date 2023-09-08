from transformers import ViTFeatureExtractor
import torch
from datasets import load_metric
import numpy as np


repo_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(repo_id)


def process_example(imagen):
  inputs = feature_extractor(imagen["image"], return_tensors="pt")
  inputs["labels"] = imagen["label"]
  return inputs


def transform(example_batch):
  inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["labels"]= example_batch["label"]

  return inputs

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x['pixel_values'] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])

    }

metric = load_metric("accuracy")
def compute_metrics(prediction):
    return metric.compute(predictions=np.argmax(prediction.predictions, axis=1), references=prediction.label_ids)


