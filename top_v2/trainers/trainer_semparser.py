#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_trainer.py

import json
import os
import re
import argparse
import logging
from collections import namedtuple
from tracemalloc import start
from typing import Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD

from custom_datasets.RINE import RINEDataset
from custom_datasets.RINE_predictor_dataset import RINEPredictorDataset
from custom_datasets.collate_functions import collate_to_max_length
from models.semparser_diff_layer_with_grammar import SemparserRINE
from models.model_config import RobertaQueryNerConfig
from utils.utils import get_parser, set_random_seed, invert_mask, label_masks_based_on_grammars
from pre_processing.tree import Tree, get_node_info


class BertLabeling(pl.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.default_root_dir, "eval_result_log.txt"), level=logging.INFO)
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            print("*=" * 50)
            print(self.args)
            logging.basicConfig(format=format, filename=os.path.join(self.args.default_root_dir, "eval_test.txt"), level=logging.INFO)

        print("Set random_seed to:", int(args.seed))
        set_random_seed(args.seed)
        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir
        weight_sum = args.weight_start + args.weight_end + args.weight_type
        weight_start = args.weight_start
        weight_end = args.weight_end
        weight_type = args.weight_type
        label2id_path = os.path.join(self.data_dir, f"{self.args.domain}_label2id.json")
        self.label2id = json.load(open(label2id_path, "r"))
        self.label2id["[EOP]"] = len(self.label2id.values())
        self.num_label_types = len(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
        grammar_path = os.path.join(self.data_dir, f"extracted_grammar_{self.args.domain}_{self.args.spis}spis.json")
        self.grammars = json.load(open(grammar_path, "r"))

        bert_config = RobertaQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         mrc_dropout=args.mrc_dropout,
                                                         classifier_act_func = args.classifier_act_func,
                                                         classifier_intermediate_hidden_size=args.classifier_intermediate_hidden_size,
                                                         loss_weight=(weight_start, weight_end, weight_type),
                                                         ignore_index=args.ignore_index,
                                                         possible_labels_span=args.possible_labels_span,
                                                         num_label_types=self.num_label_types,
                                                         span_loss_candidates=args.span_loss_candidates)

        print("Model name:", args.semparser_model)
        print("Max seq length:", args.max_length)
        
        self.model = eval(args.semparser_model).from_pretrained(args.bert_config_dir,
                                                  config=bert_config)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        
        self.flat_ner = args.flat
        self.optimizer = args.optimizer
        self.span_loss_candidates = args.span_loss_candidates
        self.ignore_index = args.ignore_index
        self.max_pred_step = args.max_pred_step
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
        self.pad_id = 1
        self.test_log = []
        self.dev_log = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="bert dropout rate")
        parser.add_argument("--classifier_act_func", type=str, default="gelu")
        parser.add_argument("--classifier_intermediate_hidden_size", type=int, default=1024)
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_type", type=float, default=1.0)
        parser.add_argument("--flat", action="store_true", help="is flat ner")
        parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "pred_gold_random", "gold"],
                            default="all", help="Candidates used to compute span loss")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "adam"], default="adam",
                            help="loss type")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle", )
        parser.add_argument("--lr_mini", type=float, default=-1)
        parser.add_argument("--ignore_index", type=int, default=-100)
        parser.add_argument("--possible_labels_span", type=int, default=5)
        parser.add_argument("--max_pred_step", type=int, default=20)
        parser.add_argument("--semparser_model", type=str, default="SemparserRINE")        
        parser.add_argument("--domain", type=str, default="")
        parser.add_argument("--spis", type=str, default="")
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "adam":
            print("Betas:", (0.9, 0.98))
            print("self.args.lr:", self.args.lr)
            print("self.args.adam_epsilon:", self.args.adam_epsilon)
            print("self.args.weight_decay:", self.args.weight_decay)
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          betas=(0.9, 0.98),
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / 5
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, label_types, label_mask = batch

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != self.pad_id).long()
        _, total_loss = self.model(tokens, token_type_ids, attention_mask, start_label_mask, \
                                                            end_label_mask, label_mask, start_labels, end_labels, label_types)

        tf_board_logs[f"train_loss"] = total_loss
        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}
        self.model.eval()

        # Extract data from batch
        # try:
        sentence, starts, ends, entities, org_label = batch["tokens"], batch["starts"], batch["ends"], batch["entities"], batch["org_label"]
        start_labels = [x.item() for x in starts]
        end_labels = [x.item() for x in ends]
        entity_labels = [x.item() for x in entities]
        start_preds, end_preds, entity_preds = [], [], []

        # Get predictions
        cur_step, cur_label = 0, None
        context = sentence[0]
        org_label = org_label[0]     
        # print("----------------------------")
        while (cur_step < self.max_pred_step) and (cur_label != "[EOP]"):
            cur_step += 1
            pred = context
            # Get token ids
            # print("context:", context)
            # print("cur_step:", cur_step)

            sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True, max_length=512)
            input_ids = torch.LongTensor(sample_tokens.input_ids).unsqueeze(0).to( "cuda:0")
            type_ids = torch.LongTensor(sample_tokens.token_type_ids).unsqueeze(0).to("cuda:0")
            attention_mask = (input_ids != self.pad_id).long()

            # Get predictions
            with torch.no_grad():
                start_logits, end_logits, entity_logit = self.model.predict(input_ids, type_ids, attention_mask)
            start_pred = torch.argmax(start_logits, dim=1).item()
            end_masks = [id >= start_pred  for id in range(start_logits.shape[1])]
            end_masks = torch.FloatTensor([end_masks]).to(end_logits.device)
            inf_masks = torch.full(end_logits.shape, -1000).to(end_logits.device)
            end_logits = end_logits * end_masks + invert_mask(end_masks) * inf_masks
            end_pred = torch.argmax(end_logits, dim=1).item()

            # Check if it is the last step
            if end_pred == 0  and start_pred == 0:
                cur_label = "[EOP]"
                continue

            # Get tokens 
            tokens = input_ids[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)
            tokens = tokens[1:-1]
            tmp_start_pred = start_pred-1 if start_pred > 0 else 0
            tmp_end_pred = end_pred-1 if end_pred > 0 else 0

            # Check candidates by grammars
            entity_mask = label_masks_based_on_grammars(tokens, tmp_start_pred, self.grammars, self.label2id)
            entity_logit = entity_logit.masked_fill(entity_mask==0, float(-1000))
            entity_pred = torch.argmax(entity_logit, dim=1).item()
            cur_label = self.id2label[entity_pred]

            end_token = "Ġ]"
            start_token = f"Ġ[{cur_label}" if tmp_start_pred > 0 else f"[{cur_label}Ġ"
            tokens.insert(tmp_end_pred+1, end_token)
            tokens.insert(tmp_start_pred, start_token)
            context = "".join(tokens).replace("Ġ", " ")

            if cur_label != "[EOP]":
                start_preds.append(start_pred)
                end_preds.append(end_pred)
                entity_preds.append(entity_pred)

        # Check if the prediction is correct
        self.dev_log.append({
            "sentence": sentence[0],
            "pred": pred,
            "label": org_label,
            "start_preds": start_preds,
            "start_labels": start_labels,
            "end_preds": end_preds,
            "end_labels": end_labels,
            "entity_preds": entity_preds,
            "entity_labels": entity_labels
        })
        if start_labels == start_preds and end_labels == end_preds and entity_labels == entity_preds:
            output["exact_match"] = {"correct": 1, "total": 1}
        else:
            output["exact_match"] = {"correct": 0, "total": 1}
        # except:
        #     output["exact_match"] = {"correct": 0, "total": 1}
        return output

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}

        exact_match_correct = sum([x['exact_match']['correct'] for x in outputs])
        exact_match_total = sum([x['exact_match']['total'] for x in outputs])
        exact_match_acc = exact_match_correct / exact_match_total
        tensorboard_logs[f"exact_match_acc"] = exact_match_acc
        self.result_logger.info(f"EVAL INFO -> exact_match_span_acc is: {exact_match_acc}.")

        json.dump(self.dev_log, open(os.path.join(self.args.default_root_dir, "dev_logs", f"dev_log_epoch_{self.current_epoch}.json"), "w"), ensure_ascii=False, indent=2)
        self.dev_log = []
        return {'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        self.model.eval()
        output = {}

        # Extract data from batch
        sentence, starts, ends, entities, org_label = batch["tokens"], batch["starts"], batch["ends"], batch["entities"], batch["org_label"]
        start_labels = [x.item() for x in starts]
        end_labels = [x.item() for x in ends]
        entity_labels = [x.item() for x in entities]
        start_preds, end_preds, entity_preds = [], [], []

        # Get predictions
        cur_step, cur_label = 0, None
        context = sentence[0]
        org_label = org_label[0]            
        # print("----------------------------")
        pred = ""
        while (cur_step < self.max_pred_step) and (cur_label != "[EOP]"):
            cur_step += 1

            # Get token ids
            # print("context:", context)
            # print("cur_step:", cur_step)

            pred = context
            sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True, max_length=512)
            input_ids = torch.LongTensor(sample_tokens.input_ids).unsqueeze(0).to( "cuda:0")
            type_ids = torch.LongTensor(sample_tokens.token_type_ids).unsqueeze(0).to("cuda:0")
            attention_mask = (input_ids != self.pad_id).long()

            # Get predictions
            with torch.no_grad():
                start_logits, end_logits, entity_logit = self.model.predict(input_ids, type_ids, attention_mask)
            start_pred = torch.argmax(start_logits, dim=1).item()
            end_masks = [id >= start_pred  for id in range(start_logits.shape[1])]
            end_masks = torch.FloatTensor(end_masks).to(end_logits.device)
            inf_masks = torch.full(end_logits.shape, -1000).to(end_logits.device)
            end_logits = end_logits * end_masks + invert_mask(end_masks) * inf_masks
            end_pred = torch.argmax(end_logits, dim=1).item()

            # Check if it is the last step
            if end_pred == 0  and start_pred == 0:
                cur_label = "[EOP]"
                continue

            # Get tokens 
            tokens = input_ids[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)
            tokens = tokens[1:-1]
            tmp_start_pred = start_pred-1 if start_pred > 0 else 0
            tmp_end_pred = end_pred-1 if end_pred > 0 else 0

            # Check candidates by grammars
            entity_mask = label_masks_based_on_grammars(tokens, tmp_start_pred, self.grammars, self.label2id)
            entity_logit = entity_logit.masked_fill(entity_mask==0, float(-1000))
            entity_pred = torch.argmax(entity_logit, dim=1).item()
            cur_label = self.id2label[entity_pred]

            end_token = "Ġ]"
            start_token = f"Ġ[{cur_label}" if tmp_start_pred > 0 else f"[{cur_label}Ġ"
            tokens.insert(tmp_end_pred+1, end_token)
            tokens.insert(tmp_start_pred, start_token)
            context = "".join(tokens).replace("Ġ", " ")

            if cur_label != "[EOP]":
                start_preds.append(start_pred)
                end_preds.append(end_pred)
                entity_preds.append(entity_pred)

        # Check if the prediction is correct
        self.test_log.append({
            "sentence": sentence[0],
            "pred": pred,
            "label": org_label,
            "start_preds": start_preds,
            "start_labels": start_labels,
            "end_preds": end_preds,
            "end_labels": end_labels,
            "entity_preds": entity_preds,
            "entity_labels": entity_labels
        })
        try:
            gold_tree = Tree(pred)
            pred_info = get_node_info(gold_tree)
            start_preds = [item[0] for item in pred_info]
            end_preds = [item[1] for item in pred_info]
            entity_preds = [item[2] for item in pred_info]

            gold_tree = Tree(org_label)
            labe_info = get_node_info(gold_tree)
            start_labels = [item[0] for item in labe_info]
            end_labels = [item[1] for item in labe_info]
            entity_labels = [item[2] for item in labe_info]

            if start_labels == start_preds and end_labels == end_preds and entity_labels == entity_preds:
                output["exact_match"] = {"correct": 1, "total": 1}
            else:
                output["exact_match"] = {"correct": 0, "total": 1}
        except:
            output["exact_match"] = {"correct": 0, "total": 1}

        return output

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, Tensor]]:
        tensorboard_logs = {}

        exact_match_correct = sum([x['exact_match']['correct'] for x in outputs])
        exact_match_total = sum([x['exact_match']['total'] for x in outputs])
        exact_match_acc = exact_match_correct / exact_match_total
        tensorboard_logs[f"exact_match_acc"] = exact_match_acc
        self.result_logger.info(f"TEST INFO -> exact_match_span_acc is: {exact_match_acc}.")
        
        json.dump(self.test_log, open(os.path.join(self.args.default_root_dir, "test_logs", f"test_log.json"), "w"), ensure_ascii=False, indent=2)
        self.test_log = []
        return {'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        if prefix == "test":
            json_path = os.path.join(self.data_dir, f"{self.args.domain}-mrc-ner.{prefix}")
        else:
            json_path = os.path.join(self.data_dir, f"{self.args.domain}_{self.args.spis}spis-mrc-ner.{prefix}")
        tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
        label2id_path = os.path.join(self.data_dir, f"{self.args.domain}_label2id.json")
        grammar_path = os.path.join(self.data_dir, f"extracted_grammar_{self.args.domain}_{self.args.spis}spis.json")

        print("grammar_path", grammar_path)
        print("label2id_path", label2id_path)
        if prefix == "train":
            dataset = RINEDataset(json_path=json_path,
                                    label2id_path=label2id_path,
                                    grammar_path=grammar_path,
                                    tokenizer=tokenizer,
                                    max_length=self.args.max_length
                                )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.workers,
                shuffle=False,
                collate_fn= lambda b: collate_to_max_length(b, ignore_index=self.ignore_index, pad_id=self.pad_id)
            )
        else:
            dataset = RINEPredictorDataset(json_path=json_path, label2id_path=label2id_path, tokenizer=tokenizer)
            dataloader = DataLoader(dataset, batch_size=1)
            
        return dataloader

def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = False):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"exact_match_acc reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("exact_match_acc reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")

        if current_f1 >= best_f1_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLabeling(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=args.max_keep_ckpt,
        verbose=True,
        monitor="exact_match_acc",
        period=-1,
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        default_root_dir=args.default_root_dir
    )

    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.default_root_dir, )
    model.result_logger.info("=&" * 20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.result_logger.info("=&" * 20)


if __name__ == '__main__':
    main()