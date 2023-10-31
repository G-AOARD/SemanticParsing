from tree import Tree, Token
import json
from tqdm import tqdm
import unidecode

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"

def get_grammar_rule(cur_node, grammar):
    for child in cur_node.children:
        if type(child) != Token:
            if cur_node.label not in grammar:
                grammar[cur_node.label] = set()
            grammar[cur_node.label].add(child.label)
        get_grammar_rule(child, grammar)
    
    # add empty rule to grammar if this node contains all tokens
    if type(cur_node) != Token and len([child for child in cur_node.children if type(child) != Token]) == 0:
        if cur_node.label not in grammar:
            grammar[cur_node.label] = set()
        # grammar[cur_node.label].add("META-IN:EOG")
    return grammar
    
    
def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    domains = ["weather", "reminder"]
    spis_ratios = [25, 500]
    for domain in domains:
        for spis_ratio in spis_ratios:
            # grammar inductive
            grammar = {}
            with open( f"data/raw/{domain}_train_{spis_ratio}spis.tsv" ) as gold_file:
                for gold_line in tqdm(gold_file):
                    data = gold_line.split("\t")
                    label_line =  unidecode.unidecode(data[2])
                    if BRACKET_OPEN not in label_line:
                        continue
                    gold_tree = Tree(label_line)

                    grammar = get_grammar_rule(gold_tree.root, grammar)
                        
            # dump GRAMMAR when parse data
            for k, v in grammar.items():
                grammar[k] = list(v)
            write_json(grammar, f"data/extracted_grammar_{domain}_{spis_ratio}spis.json")

main()