import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from utils import convert_predictions_to_label_sequence


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        loss_calculator,
        metrics_calculator,
        output_dir,
        training_args,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_calculator = loss_calculator
        self.metrics_calculator = metrics_calculator
        self.output_dir = output_dir
        self.training_args = training_args
        self.early_stopping_count = 0
        self.writer = None
        if training_args.is_master_process:
            self.writer = SummaryWriter(output_dir)
        self.device = None
        for _, p in model.named_parameters():
            self.device = p.device
            break
        

    def train(self):
        args = self.training_args
        num_train_batches = len(self.train_dataloader)

        best_weighted_f1 = -1
        best_micro_f1 = -1
        best_macro_f1 = -1

        best_weighted_p = -1
        best_micro_p = -1
        best_macro_p = -1

        best_weighted_r = -1
        best_micro_r = -1
        best_macro_r = -1

        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):

            self.model.train()
            train_loss = 0
            train_entity_loss = 0
            train_relation_loss = 0
            self.metrics_calculator.clear()

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=num_train_batches, leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = self.model(batch)
                loss = outputs["loss"]
                entity_loss = outputs["entity_loss"]
                relation_loss = outputs["relation_loss"]
                attention_mask = batch["attention_mask"].detach().cpu()

                loss.backward()
                torch.cuda.empty_cache()
                if (batch_idx + 1) % args.grad_accumulation_steps == 0 or (batch_idx + 1) == num_train_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    if self.scheduler is not None:
                        self.scheduler.step()

                train_loss += loss.item()
                train_entity_loss += entity_loss.item()
                train_relation_loss += relation_loss.item()

                entity_predictions = outputs["entity_prediction"]
                entity_predictions = convert_predictions_to_label_sequence(
                    predictions=entity_predictions,
                    id2label=args.id2label,
                    mask=attention_mask,
                    add_special_tokens=args.add_special_tokens,
                )
                entity_references = convert_predictions_to_label_sequence(
                    predictions=batch["entity_labels"].detach().cpu().tolist(),
                    id2label=args.id2label,
                    mask=attention_mask,
                    add_special_tokens=args.add_special_tokens,
                )
                
                relation_predictions = outputs["relation_prediction"].detach().cpu()
                relation_references = batch["relation_labels"].detach().cpu()
                
                self.metrics_calculator.add_batch(
                    entity_predictions=entity_predictions,
                    entity_references=entity_references,
                    relation_predictions=relation_predictions,
                    relation_references=relation_references,
                    mask=attention_mask,
                )
                
                if self.writer is not None:
                    global_step = num_train_batches * epoch + batch_idx
                    if (global_step + 1) % args.write_step == 0:
                        self.writer.add_scalar("Loss/Step/Train/loss", loss.item(), global_step)
                        self.writer.add_scalar("Loss/Step/Train/entity_loss", entity_loss.item(), global_step)
                        self.writer.add_scalar("Loss/Step/Train/relation_loss", relation_loss, global_step)

                        if self.scheduler is not None:
                            cur_lr = self.scheduler.get_last_lr()[0]
                            self.writer.add_scalar("LearningRate-Step", cur_lr, global_step=global_step)
            
            train_metrics = self.metrics_calculator.compute()
            eval_result = self.evaluate()

            if self.training_args.training_mode == "ddp":
                train_metrics_list = [0 for _ in range(self.training_args.world_size)]
                eval_outputs_list = [0 for _ in range(self.training_args.world_size)]

                dist.all_gather_object(eval_outputs_list, eval_result)
                dist.all_gather_object(train_metrics_list, train_metrics)
            
                train_metrics = self.metrics_calculator.compute_metrics_list(train_metrics_list)

                eval_loss_list = [x["loss"] for x in eval_outputs_list]
                eval_loss = sum(eval_loss_list) / len(eval_loss_list)
                eval_entity_loss_list = [x["entity_loss"] for x in eval_outputs_list]
                eval_entity_loss = sum(eval_entity_loss_list) / len(eval_entity_loss_list)
                eval_relation_loss_list = [x["relation_loss"] for x in eval_outputs_list]
                eval_relation_loss = sum(eval_relation_loss_list) / len(eval_relation_loss_list)

                eval_metrics_list = [x["metrics"] for x in eval_outputs_list]
                eval_metrics = self.metrics_calculator.compute_metrics_list(eval_metrics_list)

                eval_result = {
                    "loss": eval_loss,
                    "entity_loss": eval_entity_loss,
                    "relation_loss": eval_relation_loss, 
                    "metrics": eval_metrics
                }

            eval_metrics = eval_result["metrics"]
            eval_overall_metrics = eval_metrics["entity"]["overall"]

            weighted_p = eval_overall_metrics["weighted-precision"]
            weighted_r = eval_overall_metrics["weighted-recall"]
            weighted_f1 = eval_overall_metrics["weighted-f1"]

            micro_p = eval_overall_metrics["micro-precision"]
            micro_r = eval_overall_metrics["micro-recall"]
            micro_f1 = eval_overall_metrics["micro-f1"]

            macro_p = eval_overall_metrics["macro-precision"]
            macro_r = eval_overall_metrics["macro-recall"]
            macro_f1 = eval_overall_metrics["macro-f1"]

            save_name = f"model_epoch_{epoch}_wf1_{weighted_f1:.4f}_wp_{weighted_p:.4f}_wr_{weighted_r:.4f}_mif1_{micro_f1:.4f}_mip_{micro_p:.4f}_mir_{micro_r:.4f}_maf1_{macro_f1:.4f}_map_{macro_p:.4f}_mar_{macro_r:.4f}.pth"
            save_path = os.path.join(self.output_dir, save_name)
            if hasattr(self.model, "module"):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.training_args.stopping_metric_type == "precision":
                if (weighted_p > best_weighted_p) or (micro_p > best_micro_p) or (macro_p > best_macro_p):
                    
                    if weighted_p > best_weighted_p:
                        best_weighted_p = weighted_p
                    
                    if micro_p > best_micro_p:
                        best_micro_p = micro_p
                    
                    if macro_p > best_macro_p:
                        best_macro_p = macro_p
                    
                    self.early_stopping_count = 0
                    if self.training_args.is_master_process:
                        torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            elif self.training_args.stopping_metric_type == "recall":
                if (weighted_r > best_weighted_r) or (micro_r > best_micro_r) or (macro_r > best_macro_r):
                    if weighted_r > best_weighted_r:
                        best_weighted_r = weighted_r
                    if micro_r > best_micro_r:
                        best_micro_r = micro_r
                    if macro_r > best_macro_r:
                        best_macro_r = macro_r

                    self.early_stopping_count = 0
                    if self.training_args.is_master_process:
                        torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if (weighted_f1 > best_weighted_f1) or (micro_f1 > best_micro_f1) or (macro_f1 > best_macro_f1):
                    if weighted_f1 > best_weighted_f1:
                        best_weighted_f1 = weighted_f1
                    if micro_f1 > best_micro_f1:
                        best_micro_f1 = micro_f1
                    if macro_f1 > best_macro_f1:
                        best_macro_f1 = macro_f1

                    self.early_stopping_count = 0
                    if self.training_args.is_master_process:
                        torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if not self.training_args.is_master_process:
                continue

            train_loss /= num_train_batches
            train_entity_loss /= num_train_batches
            train_relation_loss /= num_train_batches
            self.writer.add_scalar("Loss/Epoch/Train/loss", train_loss, epoch)
            self.writer.add_scalar("Loss/Epoch/Train/entity_loss", train_entity_loss, epoch)
            self.writer.add_scalar("Loss/Epoch/Train/relation_loss", train_relation_loss, epoch)

            train_entity_category_metrics = train_metrics["entity"]["category"]
            train_entity_overall_metrics = train_metrics["entity"]["overall"]
            for ent_type, metrics in train_entity_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Train-Entity/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in train_entity_overall_metrics.items():
                self.writer.add_scalar(f"Train-Entity-Overall/{metric_type}", value, global_step=epoch)

            train_relation_category_metrics = train_metrics["relation"]["category"]
            train_relation_overall_metrics = train_metrics["relation"]["overall"]
            for ent_type, metrics in train_relation_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Train-Relation/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in train_relation_overall_metrics.items():
                self.writer.add_scalar(f"Train-Relation-Overall/{metric_type}", value, global_step=epoch)


            eval_loss = eval_result["loss"]
            eval_entity_loss = eval_result["entity_loss"]
            eval_relation_loss = eval_result["relation_loss"]
            self.writer.add_scalar("Loss/Epoch/Eval/loss", eval_loss, epoch)
            self.writer.add_scalar("Loss/Epoch/Eval/entity_loss", eval_entity_loss, epoch)
            self.writer.add_scalar("Loss/Epoch/Eval/relation_loss", relation_loss, epoch)

            eval_metrics = eval_result["metrics"]
            eval_entity_category_metrics = eval_metrics["entity"]["category"]
            eval_entity_overall_metrics = eval_metrics["entity"]["overall"]
            for ent_type, metrics in eval_entity_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Eval-Entity/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in eval_entity_overall_metrics.items():
                self.writer.add_scalar(f"Eval-Entity-Overall/{metric_type}", value, global_step=epoch)

            eval_relation_category_metrics = eval_metrics["relation"]["category"]
            eval_relation_overall_metrics = eval_metrics["relation"]["overall"]
            for ent_type, metrics in eval_relation_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Eval-Relation/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in eval_relation_overall_metrics.items():
                self.writer.add_scalar(f"Eval-Relation-Overall/{metric_type}", value, global_step=epoch)


    def evaluate(self):
        args = self.training_args
        self.model.eval()
        self.metrics_calculator.clear()

        eval_loss = 0
        eval_entity_loss = 0
        eval_relation_loss = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = self.model(batch)
                outputs = self.model(batch)
                loss = outputs["loss"]
                entity_loss = outputs["entity_loss"]
                relation_loss = outputs["relation_loss"]
                attention_mask = batch["attention_mask"].detach().cpu()

                eval_loss += loss.item()
                eval_entity_loss += entity_loss.item()
                eval_relation_loss += relation_loss.item()

                entity_predictions = outputs["entity_prediction"]
                entity_predictions = convert_predictions_to_label_sequence(
                    predictions=entity_predictions,
                    id2label=args.id2label,
                    mask=attention_mask,
                    add_special_tokens=args.add_special_tokens,
                )
                entity_references = convert_predictions_to_label_sequence(
                    predictions=batch["entity_labels"].detach().cpu().tolist(),
                    id2label=args.id2label,
                    mask=attention_mask,
                    add_special_tokens=args.add_special_tokens,
                )
                
                relation_predictions = outputs["relation_prediction"].detach().cpu()
                relation_references = batch["relation_labels"].detach().cpu()
                
                self.metrics_calculator.add_batch(
                    entity_predictions=entity_predictions,
                    entity_references=entity_references,
                    relation_predictions=relation_predictions,
                    relation_references=relation_references,
                    mask=attention_mask,
                )

                
        eval_loss /= len(self.eval_dataloader)
        eval_entity_loss /= len(self.eval_dataloader)
        eval_relation_loss /= len(self.eval_dataloader)
        metrics = self.metrics_calculator.compute()

        eval_result = {
            "loss": eval_loss,
            "entity_loss": eval_entity_loss,
            "relation_loss": eval_relation_loss,
            "metrics": metrics,
        }

        return eval_result