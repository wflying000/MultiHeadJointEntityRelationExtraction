import os
import json
import time
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from model import JointModel
from trainer import Trainer
from utils import GeneralConfig, SequenceTaggingDecoder
from dataset import JointDataset, get_data
from metrics import EntityRelationMetrics


def get_args():
    args_parser = ArgumentParser()
    
    args_parser.add_argument("--train_data_path", default="../data/train_data.json", type=str)
    args_parser.add_argument("--eval_data_path", default="../data/dev_data.json", type=str)
    args_parser.add_argument("--ner_rel_info_path", default="../data/ner_rel_info.json", type=str)
    args_parser.add_argument("--pretrained_model_path", default="../../pretrained_model/voidful/albert_chinese_tiny/", type=str)
    args_parser.add_argument("--max_length", default=256, type=int)
    args_parser.add_argument("--train_batch_size", default=16, type=int)
    args_parser.add_argument("--eval_batch_size", default=16, type=int)
    args_parser.add_argument("--entity_label_dim", default=12, type=int)
    args_parser.add_argument("--relation_embed_dim", default=64, type=int)
    args_parser.add_argument("--rel_weight", default=50.0, type=float)
    args_parser.add_argument("--rel_pos_weight", default=20.0, type=float)
    args_parser.add_argument("--relation_threshold", default=0.5, type=float)
    args_parser.add_argument("--cuda_idx", default=0, type=int)
    args_parser.add_argument("--lr", default=1e-3, type=float)
    args_parser.add_argument("--num_epochs", default=20, type=int)
    args_parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    args_parser.add_argument("--add_special_tokens", default=False, type=bool)
    args_parser.add_argument("--patience", default=10, type=int)
    args_parser.add_argument("--output_dir", default="../outputs/", type=str)

    args = args_parser.parse_args()

    return args


def train(args):
    
    train_data = get_data(args.train_data_path)
    eval_data = get_data(args.eval_data_path)
    ner_rel_info = json.load(open(args.ner_rel_info_path))
    entity_types = ner_rel_info["entity_types"]
    ent2id = ner_rel_info["ent2id"]
    id2ent = {v: k for k, v in ent2id.items()}
    relation_types = ner_rel_info["relation_types"]
    rel2id = ner_rel_info["rel2id"]
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    pretrained_model_config = AutoConfig.from_pretrained(args.pretrained_model_path)
    hidden_size = pretrained_model_config.hidden_size

    train_dataset = JointDataset(
        data=train_data,
        max_length=args.max_length,
        tokenizer=tokenizer,
        ent2id=ent2id,
        rel2id=rel2id,
    )

    eval_dataset = JointDataset(
        data=eval_data,
        max_length=args.max_length,
        tokenizer=tokenizer,
        ent2id=ent2id,
        rel2id=rel2id,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    model_config = GeneralConfig(
        pretrained_model_path=args.pretrained_model_path,
        hidden_size=hidden_size,
        num_entity_labels=len(ent2id),
        entity_label_embed_dim=args.entity_label_dim,
        num_relation_labels=len(rel2id),
        relation_embed_dim=args.relation_embed_dim,
        rel_weight=args.rel_weight,
        rel_pos_weight=args.rel_pos_weight,
        relation_threshold=args.relation_threshold,
    )

    model = JointModel(model_config)
    device = torch.device(f"cuda:{args.cuda_idx}")
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=8, min_lr=1e-6, verbose=True
    )

    training_args = GeneralConfig(
        is_master_process=True,
        num_epochs=args.num_epochs,
        grad_accumulation_steps=args.grad_accumulation_steps,
        id2label=id2ent,
        add_special_tokens=args.add_special_tokens,
        training_mode="single",
        world_size=1,
        stopping_metric_type="f1",
        patience=args.patience,
        write_step=1,
    )

    seq_tag_decoder = SequenceTaggingDecoder(scheme="BIO")
    metrics_calculator = EntityRelationMetrics(entity_types, id2rel, seq_tag_decoder)
    
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_dir = f"{args.output_dir}/{now_time}/"
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=None,
        loss_calculator=None,
        metrics_calculator=metrics_calculator,
        output_dir=output_dir,
        training_args=training_args,
    )

    trainer.train()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    import sys
    os.chdir(sys.path[0])
    main()


    


