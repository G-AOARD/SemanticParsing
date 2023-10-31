#!/usr/bin/env python3

from typing import List, Optional, Tuple

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"
PREFIX_INTENT = "IN:"
PREFIX_SLOT = "SL:"


class Node:
    """
    A generalization of Root / Intent / Slot / Token
    """
    def __init__(self, label: str) -> None:
        self.label: str = label
        self.children: List[Node] = []
        self.parent: Optional[Node] = None

    def validate_node(self) -> None:
        for child in self.children:
            child.validate_node()

    def list_nonterminals(self):
        non_terminals: List[Node] = []
        for child in self.children:
            if type(child) != Root and type(child) != Token:
                non_terminals.append(child)
                non_terminals += child.list_nonterminals()
        return non_terminals

    def get_token_indices(self) -> List[int]:
        indices: List[int] = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    indices.append(child.index)
                else:
                    indices += child.get_token_indices()
        return indices

    def get_token_span(self) -> Optional[Tuple[int, int]]:
        indices = self.get_token_indices()
        if indices:
            return (min(indices), max(indices) + 1)
        return None

    def get_flat_str_spans(self) -> str:
        str_span: str = str(self.get_token_span()) + ": "
        if self.children:
            for child in self.children:
                str_span += str(child)
        return str_span

    def __repr__(self) -> str:
        str_repr: str = ""
        if type(self) == Intent or type(self) == Slot:
            str_repr = BRACKET_OPEN
        if type(self) != Root:
            str_repr += str(self.label) + " "
        if self.children:
            for child in self.children:
                str_repr += str(child)
        if type(self) == Intent or type(self) == Slot:
            str_repr += BRACKET_CLOSE + " "
        return str_repr


class Root(Node):
    def __init__(self) -> None:
        super().__init__("ROOT")

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError(
                    "A Root's child must be an Intent or Token: " + self.label)
            elif self.parent is not None:
                raise TypeError(
                    "A Root should not have a parent: " + self.label)


class Intent(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Intent or type(child) == Root:
                raise TypeError(
                    "An Intent's child must be a slot or token: " + self.label)


class Slot(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError("An Slot's child must be an intent or token: "
                                + self.label)


class Token(Node):
    def __init__(self, label: str, index: int) -> None:
        super().__init__(label)
        self.index: int = index

    def validate_node(self) -> None:
        if len(self.children) > 0:
            raise TypeError("A Token {} can't have children: {}".format(
                self.label, str(self.children)))


class Tree:
    def __init__(self, top_repr: str) -> None:
        self.root = Tree.build_tree(top_repr)
        try:
            self.validate_tree()
        except ValueError as v:
            raise ValueError("Tree validation failed: {}".format(v))

    @staticmethod
    def build_tree(top_repr: str) -> Root:
        root = Root()
        node_stack: List[Node] = [root]
        token_count: int = 0

        for item in top_repr.split():
            if item == BRACKET_CLOSE:
                if not node_stack:
                    raise ValueError("Tree validation failed")
                node_stack.pop()

            elif item.startswith(BRACKET_OPEN):
                label: str = item[1:]
                if label.startswith(PREFIX_INTENT):
                    node_stack.append(Intent(label))
                elif label.startswith(PREFIX_SLOT):
                    node_stack.append(Slot(label))
                else:
                    raise NameError(
                        "Nonterminal label {} must start with {} or {}".format(
                            label, PREFIX_INTENT, PREFIX_SLOT))

                if len(node_stack) < 2:
                    raise ValueError("Tree validation failed")
                node_stack[-1].parent = node_stack[-2]
                node_stack[-2].children.append(node_stack[-1])

            else:
                token = Token(item, token_count)
                token_count += 1
                if not node_stack:
                    raise ValueError("Tree validation failed")
                token.parent = node_stack[-1]
                node_stack[-1].children.append(token)

        if len(node_stack) > 1:
            raise ValueError("Tree validation failed")

        return root

    def validate_tree(self) -> None:
        try:
            self.root.validate_node()
            for child in self.root.children:
                child.validate_node()
        except TypeError as t:
            raise ValueError("Failed validation for {} \n {}".format(
                self.root, str(t)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.root == other.root

    def __repr__(self) -> str:
        return repr(self.root).strip()

def recur_child_nodes(cur_node, node_ids, node_inputs, edges):
    parent_id = node_ids[-1]
    tmp_span = ""
    for child in cur_node.children:
        if type(child) == Token:
            tmp_span += " " + child.label
        else:
            # Add span node
            if tmp_span != "":
                span_id = node_ids[-1] + 1
                node_ids.append(span_id)
                node_inputs.append(tmp_span.strip())
                edges.append((parent_id, span_id))
            tmp_span = ""

            # Add intent slot node
            if type(child) == Intent:
                label = "INTENT: " + child.label.split(":")[-1].replace("_", " ").lower()
            else:
                label = "SLOT: " + child.label.split(":")[-1].replace("_", " ").lower()
            intent_slot_id = node_ids[-1] + 1
            node_ids.append(intent_slot_id)
            node_inputs.append(label)
            edges.append((parent_id, intent_slot_id))
            recur_child_nodes(child, node_ids, node_inputs, edges)

    if tmp_span != "":
        span_id = node_ids[-1] + 1
        node_ids.append(span_id)
        node_inputs.append(tmp_span.strip())
        edges.append((parent_id, span_id))

def get_graph_inputs(context):
    gold_tree = Tree(context)
    root = gold_tree.root.children[0]
    if type(root) == Token:
        node_ids, node_inputs, edges = [0], [context], []
    else:
        if type(root) == Intent:
            label = "INTENT: " + root.label.split(":")[-1].replace("_", " ").lower()
        else:
            label = "SLOT: " + root.label.split(":")[-1].replace("_", " ").lower()
        node_ids, node_inputs, edges = [0], [label], []
        recur_child_nodes(root, node_ids, node_inputs, edges)
    return node_ids, node_inputs, edges

def update_pos(start_positions, end_positions, cursor):
    new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
    new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
    return new_start_positions, new_end_positions

def create_new_input(word_level_starts, word_level_ends, word_level_labels, org_context, new_label_sent):
    # Extract new pos
    words = new_label_sent.split()
    start, end, label = 0, 0, "NONE_LABEL"
    for idx, word in enumerate(words):
        if word.startswith(BRACKET_CLOSE):
            end = max(idx - 2, 0)
        elif word.startswith(BRACKET_OPEN):
            start = idx
            label = word[1:]
    # print("=")
    # print("new_label_sent", new_label_sent)
    word_level_starts.append(start)
    word_level_ends.append(end)
    word_level_labels.append(label)

    # Create context
    context = org_context
    cur_words = context.split()
    num_entities = len(word_level_starts)
    start_positions, end_positions = word_level_starts, word_level_ends
    for idx in range(num_entities):
        cur_words.insert(end_positions[idx]+1, f"]")
        cur_words.insert(start_positions[idx], f"[{word_level_labels[idx]}")
        start_positions, end_positions = update_pos(start_positions, end_positions, start_positions[idx])
        start_positions, end_positions = update_pos(start_positions, end_positions, end_positions[idx]+1)
    
    # print("word_level_starts", word_level_starts)
    # print("word_level_ends", word_level_ends)
    # print("word_level_labels", word_level_labels)
    return word_level_starts, word_level_ends, word_level_labels, " ".join(cur_words)

def get_node_info(tree):
    node_info = list_nonterminals(tree.root, [], 1)
    return node_info

def list_nonterminals(cur_node, node_info, node_level):
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            
            tmp = get_span(child)
            node_info.append((tmp[0], tmp[1]-1,child.label))

    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            list_nonterminals(child, node_info, node_level+1)
            
    return node_info
    
def get_span(node):
    return node.get_token_span()