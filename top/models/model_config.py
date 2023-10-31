#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: model_config.py

from transformers import BertConfig, RobertaConfig


class RobertaQueryNerConfig(RobertaConfig):
    def __init__(self, **kwargs):
        super(RobertaQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")
        self.loss_weight = kwargs.get("weight_pos_span", (1.0, 1.0, 1.0))
        self.ignore_index = kwargs.get("ignore_index", -100)
        self.possible_labels_span = kwargs.get("possible_labels_span", 4)
        self.num_label_types = kwargs.get("num_slot_type_labels", 60)
        self.span_loss_candidates = kwargs.get("span_loss_candidates", "all")
