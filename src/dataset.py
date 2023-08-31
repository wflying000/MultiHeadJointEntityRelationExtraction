import json
import torch


def get_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class JointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length,
        ent2id,
        rel2id,
        add_special_tokens=False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.add_special_tokens=add_special_tokens

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, item_list):
        text_list = [x["text"] for x in item_list]
        tokenized_text = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
        )

        bsz = len(item_list)
        seq_len = len(tokenized_text.input_ids[0])

        other_id = self.ent2id["O"]
        entity_labels = torch.full((bsz, seq_len), other_id, dtype=torch.long)
        relation_labels = torch.zeros(bsz, seq_len, seq_len, dtype=torch.long)

        for idx in range(bsz):
            item = item_list[idx]
            spo_list = item["spo_list"]

            for spo in spo_list:
                spans = spo["subject_span"]
                ent_type = spo["subject_type"]
                sub_start_token_idxs = []
                for start_char_idx, end_char_idx in spans:
                    start_token_idx = tokenized_text.char_to_token(idx, start_char_idx)
                    end_token_idx = tokenized_text.char_to_token(idx, end_char_idx - 1)
                    if start_token_idx is None or end_token_idx is None:
                        continue
                    
                    for j, token_idx in enumerate(range(start_token_idx, end_token_idx + 1)):
                        if j == 0:
                            entity_labels[idx][token_idx] = self.ent2id[f"B-{ent_type}"]
                            sub_start_token_idxs.append(token_idx)
                        else:
                            entity_labels[idx][token_idx] = self.ent2id[f"I-{ent_type}"]
                
                spans = spo["object_span"]
                ent_type = spo["object_type"]
                obj_start_token_idxs = []
                for start_char_idx, end_char_idx in spans:
                    start_token_idx = tokenized_text.char_to_token(idx, start_char_idx)
                    end_token_idx = tokenized_text.char_to_token(idx, end_char_idx - 1)
                    if start_token_idx is None or end_token_idx is None:
                        continue
                    
                    for j, token_idx in enumerate(range(start_token_idx, end_token_idx + 1)):
                        if j == 0:
                            entity_labels[idx][token_idx] = self.ent2id[f"B-{ent_type}"]
                            obj_start_token_idxs.append(token_idx)
                        else:
                            entity_labels[idx][token_idx] = self.ent2id[f"I-{ent_type}"]
                
                rel_type = spo["predicate"]
                rel_type_id = self.rel2id[rel_type]
                for sub_idx in sub_start_token_idxs:
                    for obj_idx in obj_start_token_idxs:
                        relation_labels[idx, sub_idx, obj_idx] = rel_type_id
            
            # offset_mapping = tokenized_text.offset_mapping[idx]
            # for token_idx in range(seq_len):
            #     if offset_mapping[token_idx] == (0, 0):
            #         entity_labels[idx][token_idx] = -100 # 特殊字符不参与loss计算
            
        batch = {
            "input_ids": torch.LongTensor(tokenized_text.input_ids),
            "attention_mask": torch.ByteTensor(tokenized_text.attention_mask),
            "entity_labels": entity_labels,
            "relation_labels": relation_labels,
        }

        return batch
            
                        

def test_dataset():
    import os
    import sys
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])
    
    data_path = "../data/test_data.json"
    pretrained_model_path = "../../pretrained_model/ernie-3.0-base-zh"
    ner_rel_info_path = "../data/ner_rel_info.json"
    max_length = 256

    data = get_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    ner_rel_info = json.load(open(ner_rel_info_path))
    entity_types = ner_rel_info["entity_types"]
    ent2id = ner_rel_info["ent2id"]
    id2ent = {v: k for k, v in ent2id.items()}
    relation_types = ner_rel_info["relation_types"]
    rel2id = ner_rel_info["rel2id"]
    id2rel = {v: k for k, v in rel2id.items()}

    dataset = JointDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        ent2id=ent2id,
        rel2id=rel2id,
        add_special_tokens=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        pass
        



def test_get_data():
    import os, sys
    os.chdir(sys.path[0])
    path = "../data/predict.json"
    data = get_data(path)
    print(len(data))


if __name__ == "__main__":
    test_dataset()