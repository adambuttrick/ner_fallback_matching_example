import re
import csv
import argparse
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from unidecode import unidecode


# Normalize and split the affiliation string input to be consistent
# with format of the NER model training data
def normalize_and_split(affiliation):
    steps = [
        lambda x: x.lower(),
        lambda x: re.sub(r'<[^>]+>', '', x),  # Remove HTML tags
        lambda x: re.sub(r'[^\w\s]', '', x),  # Remove punctuation
        lambda x: re.sub(r'\s+', ' ', x).strip(),  # Remove extra spaces
        unidecode,  # Normalize unicode characters
    ]
    for step in steps:
        affiliation = step(affiliation)
    return affiliation.split()


def load_model(model_path='adambuttrick/ner-test-bert-base-uncased-finetuned-500K-AdamW-3-epoch-locations'):
    model = BertForTokenClassification.from_pretrained(model_path)
    model = model.cpu()
    return model


def load_tokenizer(model_name='bert-base-uncased'):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return tokenizer


def preprocess_input(tokenizer, affiliation):
    inputs = tokenizer(affiliation,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding="max_length",
                       truncation=True,
                       max_length=128,  # same as NER model training data
                       return_tensors="pt")
    return inputs


def conduct_inference(model, inputs):
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]
    active_logits = logits.view(-1, model.num_labels)  # Reshape logits
    flattened_predictions = torch.argmax(
        active_logits, axis=1)  # Get predicted labels from NER model
    return ids, flattened_predictions


# Postprocess the model output so as to reconstruct back into an affiliation string
def postprocess_output(tokenizer, ids, flattened_predictions, inputs):
    ids_to_labels = {0: "B-ORG", 1: "I-ORG", 2: "O", 3: "B-LOC", 4: "I-LOC"}
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i]
                         for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:  # Filter for valid tokens
            prediction.append(token_pred[1])
        else:
            continue
    # Check for invalid tag combinations (absence of B-ORG tag)
    invalid_tag_combinations = all(tag in [
                                   "I-ORG", "B-LOC", "I-LOC", "O"] for tag in prediction) and "B-ORG" not in prediction
    if invalid_tag_combinations:
        return None
    return prediction


# Reassemble the tagged entities back into an affiliation string
def construct_string(affiliation, prediction):
    entities = []
    temp_entity = []
    tags_and_strings = {}
    for i in range(len(prediction)):
        tag = prediction[i]
        tags_and_strings[i] = f'{tag}: {affiliation[i]}'

        if tag in ["B-ORG", "I-ORG", "B-LOC", "I-LOC"]:
            if temp_entity and tag in ["B-ORG", "B-LOC"]:
                entities.append(" ".join(temp_entity))
                temp_entity = []
            temp_entity.append(affiliation[i])
        else:
            if temp_entity:
                entities.append(" ".join(temp_entity))
                temp_entity = []
    if temp_entity:
        entities.append(" ".join(temp_entity))
    # Strip any commas from the tagged/extracted strings and
    # then reassemble uniformly with commas, as the affiliation
    # matching logic parses on the basis of these.
    entities = [re.sub(',', '', entity)
                for entity in entities]
    joined_prediction = ", ".join(entities)
    return joined_prediction


def ner_inference(affiliation, model, tokenizer):
    affiliation = normalize_and_split(affiliation)
    inputs = preprocess_input(tokenizer, affiliation)
    ids, flattened_predictions = conduct_inference(
        model, inputs)
    prediction = postprocess_output(
        tokenizer, ids, flattened_predictions, inputs)
    if prediction:
        joined_prediction = construct_string(
            affiliation, prediction)
        return joined_prediction
    return None
