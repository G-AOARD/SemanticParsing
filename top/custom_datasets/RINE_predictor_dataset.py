#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

class RINEPredictorDataset(Dataset):
    """
    RINE Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """
    def __init__(self, json_path, label2id_path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = json.load(open(label2id_path, encoding="utf-8"))
        self.label2id["[EOP]"] = len(self.label2id.values())
        self.processed_data = self.process_data(self.all_data)

    @staticmethod
    def update_pos(start_positions, end_positions, cursor):
        new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
        new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
        return new_start_positions, new_end_positions
        
    def process_data(self, data):
        result = []
        print("Processing data...")
        for item in tqdm(data):
            context = item["context"]
            start_positions = item["start_position"]
            end_positions = item["end_position"]
            labels = item["label_type"]
            cur_words = context.split()
            num_entities = len(start_positions)
            cur_item = {"tokens": [], "starts": [], "ends": [], "entities": []}
            for idx in range(num_entities):
                if idx == 0:
                    cur_item["tokens"] = " ".join(cur_words)
                    cur_item["org_label"] = item["org_label"]
                start, end = self.new_start_end_pos(" ".join(cur_words), start_positions[idx], end_positions[idx])
                cur_item["starts"].append(start)
                cur_item["ends"].append(end)
                cur_item["entities"].append(self.label2id[labels[idx]])

                # Update context and postions
                cur_words.insert(end_positions[idx]+1, f"]")
                cur_words.insert(start_positions[idx], f"[{labels[idx]}")
                start_positions, end_positions = self.update_pos(start_positions, end_positions, start_positions[idx])
                start_positions, end_positions = self.update_pos(start_positions, end_positions, end_positions[idx]+1)

            result.append(cur_item)
        return result

    def new_start_end_pos(self, context, start, end):
         # query = "Find the intents and slots of the sentence."
        
        start_positions = [start] if start != -1 else []
        end_positions = [end] if end != -1 else []

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True)
        tokens = sample_tokens.input_ids
        offsets = sample_tokens.offset_mapping

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            if token_start not in origin_offset2token_idx_start:
                origin_offset2token_idx_start[token_start] = token_idx
            if token_end not in origin_offset2token_idx_end:
                origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        return new_start_positions[0], new_end_positions[0]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
        """
        return self.processed_data[item]

def run_dataset():
    """test dataset"""
    from torch.utils.data import DataLoader

    # en datasets   
    bert_path = "/home/s2110418/Master/Semantic_parsing/mrc-for-flat-nested-ner/roberta/roberta-large"
    json_path = "/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_RINE/data/top/mrc-ner.train"
    label2id_path = "/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_RINE/data/top/label2id.json"

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    dataset = RINEPredictorDataset(json_path=json_path, label2id_path=label2id_path, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=1)

    import  numpy as np
    for batch in dataloader:
        print(batch)
        break 

# if __name__ == '__main__':
#     run_dataset()
