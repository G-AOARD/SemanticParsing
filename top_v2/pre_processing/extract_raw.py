from tree import Tree, Root, Token
import json
from tqdm import tqdm
import unidecode
import argparse

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"

def get_node_info(tree):
    node_info = list_nonterminals(tree.root, [], 1)
    return node_info

def list_nonterminals(cur_node, node_info, node_level):
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            tmp = get_span(child)
            if tmp is None:
                continue
            tmp = str(tmp[0]) + ";" + str(tmp[1]-1)
            node_info.append((tmp, node_level,child.label))

    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            list_nonterminals(child, node_info, node_level+1)
            
    return node_info
    
def get_span(node):
    return node.get_token_span()

def convert_top_format_to_context_format(gold_filename):
    result = []
    with open(gold_filename) as gold_file:
        for gold_line in tqdm(gold_file):
            data = gold_line.split("\t")
            label_line =  unidecode.unidecode(data[2])
            if BRACKET_OPEN not in label_line:
                continue
            tokens = label_line.split(" ")
            context = " ".join([token for token in tokens if not (token.startswith(BRACKET_OPEN) or token.startswith(BRACKET_CLOSE))])
            gold_tree = Tree(label_line)
            label = get_node_info(gold_tree)
            result.append({
                "context": context,
                "label": label,
                "org_label": label_line.strip('\n')
            })
    return result


def count_label(data):
    label_count = {}
    for item in data:
        for label in item["label"]:
            if label[-1] not in label_count:
                label_count[label[-1]] = 1
            else:
                label_count[label[-1]] += 1
    return label_count

def create_label2id(label_count):
    queries = {"NONE_LABEL": 0}
    for idx, item in enumerate(sorted(label_count.keys())):
        queries[item] = idx + 1
    return queries

    
def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--create_label2id', action='store_true')
    args = parser.parse_args()

    phases = ["train", "valid", "test"]
    domains = ["weather", "reminder"]
    spis_ratios = ["", "_25spis", "_500spis"]
    for domain in domains:
        tmp_result = []
        for spis_ratio in spis_ratios:
            for phase in phases:
                if phase == "test":
                    gold_filename = f"./data/raw/{domain}_test.tsv"
                else:
                    gold_filename = f"./data/raw/{domain}_{phase}{spis_ratio}.tsv"
                
                result = convert_top_format_to_context_format(gold_filename)
                tmp_result += result
                if phase == "test":
                    write_json(result, f"./data/extracted_{domain}_test.json")
                else:
                    write_json(result, f"./data/extracted_{domain}_{phase}{spis_ratio}.json")
                    
        if args.create_label2id:
            label_count = count_label(tmp_result)
            label2id = create_label2id(label_count)
            write_json(label2id, f"data/{domain}_label2id.json")

main()