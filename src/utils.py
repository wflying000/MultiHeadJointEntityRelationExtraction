import re
from typing import List, Tuple
from collections import Counter


class GeneralConfig():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def convert_predictions_to_label_sequence(predictions: List[List[int]], id2label, mask, add_special_tokens):
    bsz = len(predictions)
    seq_len = len(predictions[0])
    
    if mask is not None:
        valid_lengths = mask.sum(-1).tolist()
    else:
        valid_lengths = [seq_len for _ in range(bsz)]

    label2id = {v: k for k, v in id2label.items()}
    other_id = label2id["O"]
    if add_special_tokens:
        # 如果带上特殊字符，防止将特殊字符预测为实体
        sequences = [[other_id] + seq[1: length - 1] + [other_id] for seq, length in zip(predictions, valid_lengths)]
    else:
        sequences = [seq[:length] for seq, length in zip(predictions, valid_lengths)]

    # for i, seq in enumerate(sequences):
    #     for j, x in enumerate(seq):
    #         if x < 0:
    #             print(i, j)
    #             print(seq)

    sequences = [[id2label[x] for x in seq] for seq in sequences]

    return sequences

class SequenceTaggingDecoder():
    def __init__(self, scheme: str="BIOES", pattern: str=None, tolerate_type_num: int=1):
        """
            scheme: BIOES, BIO
            pattern: 合法的序列模式
            tolerate_type_num: 合法序列所允许的类型数量最多为多少
        """

        if pattern is not None:
            self.pattern = re.compile(pattern)
        else:
            assert scheme in ("BIOES", "BIO"), \
                f"Expeced scheme in ('BIOES', 'BIO'), bug got {scheme}"
            
            if scheme == "BIOES":
                pattern = "BI*E|S"
                self.pattern = re.compile(pattern)
            elif scheme == "BIO":
                pattern = "BI*"
                self.pattern = re.compile(pattern)
        self.tolerate_type_num = tolerate_type_num

    def decode(self, sequences: List[List[str]]) -> List[List[Tuple]]:
        results = []
        for sequence in sequences:
            res = self._decode_seq_tag(sequence)
            results.append(res)
        return results
    
    def _decode_seq_tag(self, sequence: List[str]) -> List[Tuple]:
        boundary_list = [x[0] for x in sequence]
        boundary_str = "".join(boundary_list)
        type_list = [x[2:] for x in sequence]

        candidate_spans = []
        candidata_types = []
        for x in re.finditer(self.pattern, boundary_str):
            if x is None:
                continue
            span = x.span()
            start_idx = span[0]
            end_idx = span[1] - 1
            candidate_spans.append((start_idx, end_idx))
            type_count = Counter(type_list[start_idx: end_idx + 1]).most_common()
            candidata_types.append(type_count)
        
        results = []

        for span, type_count in zip(candidate_spans, candidata_types):
            if len(type_count) <= self.tolerate_type_num:
                ent_type = type_count[0][0]
                results.append((span[0], span[1], ent_type))
        
        return results