import torch
from typing import List

def compute_category_metrics(category):

    for key in category:
        TP = category[key]["TP"]
        FP = category[key]["FP"]
        FN = category[key]["FN"]
        category[key]["support"] = TP + FN

        precision = 0
        if TP + FP != 0:
            precision = TP / (TP + FP)
        
        recall = 0
        if TP + FN != 0:
            recall = TP / (TP + FN)
        
        f1 = 0
        if precision + recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        
        category[key]["precision"] = precision
        category[key]["recall"] = recall
        category[key]["f1"] = f1
    
    return category

def compute_overall_metrics(category):
    TP = 0
    FP = 0
    FN = 0
    
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for key in category:
        TP += category[key]["TP"]
        FP += category[key]["FP"]
        FN += category[key]["FN"]
        
        macro_precision += category[key]["precision"]
        macro_recall += category[key]["recall"]
        macro_f1 += category[key]["f1"]
        
        num = category[key]["TP"] + category[key]["FN"]
        weighted_precision += category[key]["precision"] * num
        weighted_recall += category[key]["recall"] * num
        weighted_f1 += category[key]["f1"] * num
    
    macro_precision /= len(category)
    macro_recall /= len(category)
    macro_f1 /= len(category)

    support = TP + FN
    if support != 0:
        weighted_precision /= support
        weighted_recall /= support
        weighted_f1 /= support

    micro_precision = 0
    if TP + FP != 0:
        micro_precision = TP / (TP + FP)
    
    micro_recall = 0
    if TP + FN != 0:
        micro_recall = TP / (TP + FN)
    
    micro_f1 = 0
    if micro_precision + micro_recall != 0:
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    overall = {
        "micro-precision": micro_precision,
        "micro-recall": micro_recall,
        "micro-f1": micro_f1,
        "macro-precision": macro_precision,
        "macro-recall": macro_recall,
        "macro-f1": macro_f1,
        "weighted-precision": weighted_precision,
        "weighted-recall": weighted_recall,
        "weighted-f1": weighted_f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "support": support,
        "num_pred": TP + FP,
    }

    return overall



class EntityRelationMetrics():
    def __init__(self, entity_types, id2rel, decoder):
        self.entity_types = entity_types
        self.id2rel = id2rel
        self.decoder = decoder
        self.relation_types = [t for _, t in id2rel.items()]

        self._init_entity_category()
        self._init_relation_category()
    
    def clear(self):
        self._init_entity_category()
        self._init_relation_category()

    def _init_relation_category(self):
        relation_types = self.id2rel.values()
        self.relation_category = self._create_category(relation_types)

    def _init_entity_category(self):
        self.entity_category = self._create_category(self.entity_types)

    def _create_category(self, types):
        category = {}
        for t in types:
            category[t] = {"TP": 0, "FP": 0, "FN": 0}
        return category
    
    def add_batch(self, entity_predictions, entity_references,
                  relation_predictions, relation_references,
                  mask):
        self._add_batch_entity(entity_predictions, entity_references)
        self._add_batch_relation(relation_predictions, relation_references, mask)
    
    def _add_batch_entity(self, predictions: List[List[str]], references: List[List[str]]):
        predictions = self.decoder.decode(predictions)
        references = self.decoder.decode(references)
        entity_category = self._compute_entity_category_count(predictions, references)
        self._update_entity_category_count(entity_category)
    
    def _add_batch_relation(self, predictions: torch.Tensor, references: torch.Tensor, mask: torch.Tensor):
        category = self._create_category(self.relation_types)
        mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        predictions = predictions.argmax(-1)
        for rel_id, rel_type in self.id2rel.items():
            TP = ((predictions == rel_id) & (references == rel_id)).sum().item()
            FP = ((predictions == rel_id) & (references != rel_id) & mask).sum().item() # 只统计有效位置，不统计填充位置
            FN = ((predictions != rel_id) & (references == rel_id)).sum().item()
            category[rel_type]["TP"] += TP
            category[rel_type]["FP"] += FP
            category[rel_type]["FN"] += FN
        self._update_relation_category_count(category)
    
    def compute(self):
        entity_metrics = self._compute_metrics(self.entity_category)
        relation_metrics = self._compute_metrics(self.relation_category)
        metrics = {
            "entity": entity_metrics,
            "relation": relation_metrics,
        }
        return metrics
    
    def _compute_metrics(self, category):
        category = compute_category_metrics(category)
        overall = compute_overall_metrics(category)
        metrics = {
            "category": category,
            "overall": overall,
        }
        return metrics

        
    def _update_relation_category_count(self, category):
        for key in category:
            self.relation_category[key]["TP"] += category[key]["TP"]
            self.relation_category[key]["FP"] += category[key]["FP"]
            self.relation_category[key]["FN"] += category[key]["FN"]
    
    def _update_entity_category_count(self, category):
        for key in category:
            self.entity_category[key]["TP"] += category[key]["TP"]
            self.entity_category[key]["FP"] += category[key]["FP"]
            self.entity_category[key]["FN"] += category[key]["FN"]
    

    def _compute_entity_category_count(self, predictions, references):
        category = self._create_category(self.entity_types)
        new_predictions = []
        for idx, preds in enumerate(predictions):
            for span in preds:
                new_span = (idx,) + span
                new_predictions.append(new_span)

        new_references = []        
        for idx, refs in enumerate(references):
            for span in refs:
                new_span = (idx,) + span
                new_references.append(new_span)
        
        pred_set = set(new_predictions)
        true_set = set(new_references)

        tp_set = pred_set.intersection(true_set)
        fp_set = pred_set.difference(true_set)
        fn_set = true_set.difference(pred_set)

        for span in tp_set:
            ent_type = span[-1]
            category[ent_type]["TP"] += 1
        for span in fp_set:
            ent_type = span[-1]
            category[ent_type]["FP"] += 1
        for span in fn_set:
            ent_type = span[-1]
            category[ent_type]["FN"] += 1
        
        return category