#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: genia2mrc.py

import os
import json
import argparse

# Raw data format:
# {
#     "context": "我是一个中国人",
#     "label": {
#         "PER": ["0;3"],
#         "LOC": ["4;7"]
#     }
# }

def convert_file(input_file, output_file):
    """
    Convert raw data to MRC format
    """
    all_data = json.load(open(input_file))
    output = []
    for idx, data in enumerate(all_data):
        context = data["context"]
        label2positions = data["label"]
        start_position, end_position, entity_level, label_type = [], [], [], []
        for label in label2positions:
            start, end = label[0].split(";")
            start_position.append(int(start))
            end_position.append(int(end))
            entity_level.append(label[1])
            label_type.append(label[-1])

        # start_position, end_position, entity_level, label_type = zip(*sorted(zip(start_position, end_position, entity_level, label_type), key=lambda x: (x[0], x[1])))

        mrc_sample = {
            "context": context,
            "entity_level": entity_level,
            "label_type": label_type,
            "start_position": start_position,
            "end_position": end_position,
            "qas_id": idx,
            "org_label": data["org_label"]
        }
        output.append(mrc_sample)
    json.dump(output, open(output_file, "w"), ensure_ascii=False, indent=2)
    print(f"Create {len(output)} samples and save to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process ...')
    args = parser.parse_args()

    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    phases = ["train", "valid", "test"]
    domains = ["weather", "reminder"]
    spis_ratios = ["", "_25spis", "_500spis"]
    for domain in domains:
        for spis_ratio in spis_ratios:
            for phase in phases:
                if phase == "test":
                    old_file = f"./data/extracted_{domain}_test.json"
                    new_file = os.path.join(base_dir, f"{domain}-mrc-ner.test")
                else:
                    old_file = f"./data/extracted_{domain}_{phase}{spis_ratio}.json"
                    new_file = os.path.join(base_dir, f"{domain}{spis_ratio}-mrc-ner.{phase}")
                convert_file(old_file, new_file)

if __name__ == '__main__':
    main()