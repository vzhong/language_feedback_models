import os
import copy
import json
import torch
import hydra
import random
import editdistance
from pathlib import Path
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.distributions.categorical import Categorical
from lightning import pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
from lightning.pytorch import callbacks as C
from lightning.pytorch.tuner.tuning import Tuner
from torch.nn import functional as F

from build_bc_data import BCDataset


class LMAgent(pl.LightningModule):

    def __init__(self, cfg, train_dataset=None, val_dataset=None, model=None):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=['train_dataset', 'val_dataset'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model or cfg.model)
        self.tokenizer = None

    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, lm_labels=None):
        outputs = self.model(
           input_ids=input_ids,
           attention_mask=attention_mask,
           decoder_attention_mask=decoder_attention_mask,
           labels=lm_labels,
        )
        return outputs

    def training_step(self, batch, batch_idx, loss_name='train_loss'):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        self.log(loss_name, outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, loss_name='val_loss')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        max_steps = int(self.trainer.max_epochs * len(self.train_dataset) / self.hparams.batch_size)
        scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=max_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def choose_action(self, example, aggregation='mean', sample=False, eval_batch_size=100):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model)
        r = example

        admissible_actions = example['admissible_actions']
        if isinstance(admissible_actions[0], dict):
            admissible_actions = [a['key'] for a in admissible_actions]

        # make smaller batches
        all_losses = []
        all_labels = []
        for i in range(0, len(admissible_actions), eval_batch_size):
            aa = admissible_actions[i:i+eval_batch_size]
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [r['inp']] * len(aa), max_length=min(self.hparams.max_len_input, self.tokenizer.model_max_length),
                truncation=True,
                return_tensors="pt",
                padding='max_length',
            ).to(self.device)
            tokenized_targets = self.tokenizer.batch_encode_plus(
                aa, max_length=self.hparams.max_len_output,
                truncation=True,
                return_tensors="pt",
                padding='max_length',
            ).to(self.device)

            source_ids = tokenized_inputs["input_ids"]
            target_ids = tokenized_targets["input_ids"]
            src_mask = tokenized_inputs["attention_mask"]
            target_mask = tokenized_targets["attention_mask"]
            labels = copy.deepcopy(target_ids)
            labels[labels == 0] = -100
            all_labels.append(labels)

            batch = {
                "source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels,
            }

            with torch.no_grad():
                outputs = self.forward(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    decoder_attention_mask=batch['target_mask'],
                    lm_labels=batch['labels'],
                )
                losses = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1), reduction='none').reshape(outputs.logits.size(0), -1)
            all_losses.append(losses)
        losses = torch.cat(all_losses, dim=0)
        labels = torch.cat(all_labels, dim=0)
        norm_losses = losses.sum(dim=1)
        if aggregation == 'mean':
            norm_losses /= (labels != -100).sum(dim=1).float()
        scores = -norm_losses

        if sample:
            p = Categorical(F.softmax(scores, dim=0))
            max_arg = p.sample().item()
            max_score = scores[max_arg].item()
        else:
            max_score, max_arg = scores.max(0)
            max_arg = max_arg.item()

        action = admissible_actions[max_arg]
        return action, scores.tolist()

    def generate_action(self, example, num_beams=4):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model)
        r = example

        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [r['inp']], max_length=min(self.hparams.max_len_input, self.tokenizer.model_max_length),
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        ).to(self.device)
        source_ids = tokenized_inputs["input_ids"]
        src_mask = tokenized_inputs["attention_mask"]

        with torch.no_grad():
            beam_outputs = self.model.generate(
                input_ids=source_ids,
                attention_mask=src_mask,
                max_length=self.hparams.max_len_output,
                early_stopping=True,
                num_beams=num_beams,
                num_return_sequences=min(num_beams, 5),
            )
        actions = [self.tokenizer.decode(
            o,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ) for o in beam_outputs]

        admissible_actions = example['admissible_actions']
        if isinstance(admissible_actions[0], dict):
            admissible_actions = [a['key'] for a in admissible_actions]

        chosen = None
        for c in actions:
            best_x = None
            best_edit_dist = float('inf')
            for xi in admissible_actions:
                edit_dist = editdistance.eval(c, xi)
                if edit_dist < best_edit_dist:
                    best_edit_dist = edit_dist
                    best_x = xi
            if best_edit_dist <= 3:
                chosen = best_x
                break
        if chosen is None:
            chosen = random.choice(admissible_actions)

        return chosen

    @classmethod
    def run_train(cls, cfg):
        with open('config.json', 'wt') as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
        ddata = Path(cfg.ddata)
        print('loading dataset from {}'.format(ddata))
        assert ddata.exists(), 'Preprocessed dataset does not exist! Make it with build_bc_data.py!'

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        kwargs = dict(
            tokenizer=tokenizer,
            max_len_input=min(cfg.max_len_input, tokenizer.model_max_length),
            max_len_output=cfg.max_len_output,
        )

        train_dataset = BCDataset.load_many([ddata.joinpath('{}.json.bz2'.format(x)) for x in cfg.train_data.split(':')], **kwargs)
        val_dataset = BCDataset.load(ddata.joinpath('{}.json.bz2'.format(cfg.val_data)), **kwargs)

        model = cls(cfg, train_dataset=train_dataset, val_dataset=val_dataset)
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            max_epochs=cfg.train_epochs,
            val_check_interval=cfg.val_interval,
            log_every_n_steps=cfg.val_interval,
            gradient_clip_val=cfg.grad_clip,
            callbacks=[
                C.ModelCheckpoint(
                    dirpath=os.getcwd(),
                    filename='{step}-{val_loss:.5f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    save_on_train_epoch_end=False,
                ),
                C.StochasticWeightAveraging(swa_lrs=cfg.weight_averaging),
            ],
        )

        if cfg.tune_batch_size:
            tuner = Tuner(trainer)
            tuner.scale_batch_size(model, mode="binsearch")
        trainer.fit(model, ckpt_path=cfg.resume)


@hydra.main(config_path='conf', config_name='bc', version_base='1.1')
def main(cfg):
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('medium')
    LMAgent.run_train(cfg)


if __name__ == '__main__':
    main()
