import os
import bz2
import glob
import json
import torch
import hydra
import evaluate
from pathlib import Path
from omegaconf import OmegaConf
from torch.optim import AdamW
from lightning import pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
from lightning.pytorch import callbacks as C
from lightning.pytorch.tuner.tuning import Tuner
from torch.nn import functional as F

from build_reward_data import FeedbackDataset
from collections import defaultdict


F1 = evaluate.load('f1')
Precision = evaluate.load('precision')
Recall = evaluate.load('recall')


class LMReward(pl.LightningModule):

    def __init__(self, cfg, train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=['train_dataset', 'val_dataset'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model)
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
        _ = self.training_step(batch, batch_idx, loss_name='val_loss')
        pred, _ = self.choose_action(batch)
        gold = batch['label_boolean'].tolist()
        pred = [1 if x else 0 for x in pred]
        gold = [1 if x else 0 for x in gold]
        self.log('val_f1', F1.compute(predictions=pred, references=gold, average='binary', pos_label=1)['f1'])
        self.log('val_p', Precision.compute(predictions=pred, references=gold)['precision'])
        self.log('val_r', Recall.compute(predictions=pred, references=gold)['recall'])

    def predict_step(self, batch, batch_idx):
        actions, scores = self.choose_action(batch, aggregation=self.hparams.eval.aggregation)
        gen = self.generate_action(batch, num_beams=self.hparams.eval.num_beams)
        gold = self.tokenizer.batch_decode(
            batch['target_ids'],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        context = self.tokenizer.batch_decode(
            batch['source_ids'],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        preds = []
        for ctx, g, p in zip(context, gold, gen):
            preds.append(dict(context=ctx, gold=g, pred=p))
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        max_steps = int(self.trainer.max_epochs * len(self.train_dataset) / self.hparams.batch_size)
        scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=max_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    def choose_action(self, batch, aggregation='mean'):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model)

        yes_tokenized_targets = self.tokenizer.batch_encode_plus(
            [FeedbackDataset.get_output(label='Yes')] * batch['source_ids'].size(0), max_length=self.hparams.max_len_output,
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        ).to(self.device)

        no_tokenized_targets = self.tokenizer.batch_encode_plus(
            [FeedbackDataset.get_output(label='No')] * batch['source_ids'].size(0), max_length=self.hparams.max_len_output,
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        ).to(self.device)

        with torch.no_grad():
            labels = yes_tokenized_targets['input_ids']
            labels[labels == 0] = -100
            label_mask = yes_tokenized_targets['attention_mask']
            outputs = self.forward(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                decoder_attention_mask=label_mask,
                lm_labels=labels,
            )
            losses = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none').reshape(outputs.logits.size(0), -1)
            # mask = (labels != -100).float()
            # yes_loss = losses.mul(mask).sum(dim=1)
            yes_loss = losses[:, 0]

            labels = no_tokenized_targets['input_ids']
            labels[labels == 0] = -100
            label_mask = no_tokenized_targets['attention_mask']
            outputs = self.forward(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                decoder_attention_mask=label_mask,
                lm_labels=labels,
            )
            losses = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none').reshape(outputs.logits.size(0), -1)
            # mask = (labels != -100).float()
            # no_loss = losses.mul(mask).sum(dim=1)
            no_loss = losses[:, 0]

            pred = (yes_loss < no_loss).tolist()
            scores = (no_loss - yes_loss).tolist()
        return pred, scores

    def generate_action(self, batch, num_beams=4):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model)
        with torch.no_grad():
            beam_outputs = self.model.generate(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                max_length=self.hparams.max_len_output,
                early_stopping=True,
                num_beams=num_beams,
                num_return_sequences=1,
            )
            action = self.tokenizer.batch_decode(
                beam_outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        return action

    @classmethod
    def run_train(cls, cfg):
        with open('config.json', 'wt') as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
        ddata = Path(cfg.ddata)
        print('loading dataset from {}'.format(ddata))
        assert ddata.exists(), 'Preprocessed dataset does not exist! Make it with build_reward_data.py!'

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        kwargs = dict(
            tokenizer=tokenizer,
            max_len_input=min(cfg.max_len_input, tokenizer.model_max_length),
            max_len_output=cfg.max_len_output,
        )
        train_dataset = FeedbackDataset.load(ddata.joinpath('train.json.bz2'), **kwargs)
        val_dataset = FeedbackDataset.load(ddata.joinpath('val.json.bz2'), **kwargs)
        # if 'touchdown' in cfg.env:
        #     train_dataset.give_reason = False
        #     val_dataset.give_reason = False

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
                    filename='{step}-{val_loss:.5f}-{val_f1:.3f}',
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
        trainer.fit(model)

    @classmethod
    def run_pred(cls, cfg):
        ddata = Path(cfg.ddata)
        print('loading dataset from {}'.format(ddata))
        assert ddata.exists(), 'rollouts do not exist at {}!!'.format(ddata)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        kwargs = dict(
            tokenizer=tokenizer,
            max_len_input=min(cfg.max_len_input, tokenizer.model_max_length),
            max_len_output=cfg.max_len_output,
        )
        val_dataset = FeedbackDataset.load(ddata.joinpath('val.json.bz2'), **kwargs)

        fsave = list(glob.glob('*.ckpt'))[0]
        model = cls(cfg, test_dataset=val_dataset)
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        )
        predictions = trainer.predict(model, model.test_dataloader(), ckpt_path=fsave)
        with bz2.open('predictions.json.bz2', 'wt') as f:
            json.dump(predictions, f, indent=2)

    @classmethod
    def run_feedback_inference(cls, cfg):
        ddata = Path(cfg.infer_feedback)
        print('loading dataset from {}'.format(ddata))
        assert ddata.exists(), 'rollouts do not exist at {}!!'.format(ddata)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        kwargs = dict(
            tokenizer=tokenizer,
            max_len_input=min(cfg.max_len_input, tokenizer.model_max_length),
            max_len_output=cfg.max_len_output,
        )
        val_dataset, val_raw = FeedbackDataset.convert_trajectories(ddata, **kwargs)
        print('dataset size {}'.format(len(val_raw)))

        fsave = list(glob.glob('*.ckpt'))[0]
        model = cls(cfg, test_dataset=val_dataset)
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        )
        predictions = trainer.predict(model, model.test_dataloader(), ckpt_path=fsave)

        by_fname = defaultdict(list)
        flat = []
        for batch in predictions:
            flat.extend(batch)
        assert len(flat) == len(val_dataset), 'got {} predictions for {} input examples'.format(len(flat), len(val_dataset))

        for ex, p in zip(val_raw, flat):
            o = ex['orig'].copy()
            o['llm_pred'] = p['pred']
            o['step'] = ex['step']
            # if 'Yes' in p['pred']:
            #     print(p)
            #     import pdb; pdb.set_trace()
            by_fname[os.path.basename(ex['fname'])].append(o)

        for traj in by_fname.values():
            traj.sort(key=lambda x: x['step'])

        dout = 'inferred_feedback/{}'.format(cfg.eval_name)
        if not os.path.isdir(dout):
            os.makedirs(dout)

        for fname, traj in by_fname.items():
            with bz2.open('{}/{}'.format(dout, fname), 'wt') as f:
                json.dump(traj, f, indent=2)


@hydra.main(config_path='conf', config_name='lfm', version_base='1.1')
def main(cfg):
    torch.set_float32_matmul_precision('medium')
    if cfg.infer_feedback:
        LMReward.run_feedback_inference(cfg)
    else:
        if not cfg.pred_only:
            LMReward.run_train(cfg)
        LMReward.run_pred(cfg)


if __name__ == '__main__':
    main()
