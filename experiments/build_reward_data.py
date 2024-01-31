import os
import re
import bz2
import copy
import glob
import random
import ujson as json
from pathlib import Path
from tqdm import auto as tqdm
from collections import Counter
from collections import defaultdict
from torch.utils.data import Dataset


class FeedbackDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_len_input=None, max_len_output=None, give_reason=True, **kwargs):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.aux = kwargs
        self.give_reason = give_reason

    def save(self, fout, **kwargs):
        if isinstance(fout, str):
            fout = Path(fout)
        if not fout.parent.is_dir():
            os.makedirs(fout.parent.as_posix())
        fopen = bz2.open if fout.as_posix().endswith('.bz2') else open
        with fopen(fout, 'wt') as f:
            json.dump(dict(data=self.data, **kwargs), f, indent=2)

    @classmethod
    def load(cls, fout, **kwargs):
        fopen = bz2.open if fout.as_posix().endswith('.bz2') else open
        with fopen(fout) as f:
            x = json.load(f)
            x.update(kwargs)
        return cls(**x)

    @classmethod
    def make(cls, data, p_pos=0.5, force_nps=False, desc='Creating feedback dataset'):
        if force_nps:
            print('Loading spacy')
            import spacy
            nlp = spacy.load("en_core_web_lg")

        proc = []

        task_re = re.compile(r'Task: ([^\n]+)\n')
        before_re = re.compile(r'Before: ([^\n]+)\n')
        step_re = re.compile(r'Step (\d+)\nYour action: ([^\n]+?)\nResult: (.+?)\-\-\-', re.DOTALL)
        feedback_re = re.compile(r'\- ([^:]+): ([^\n]+)\n')
        feedback_step_re = re.compile(r'\d+')

        for ex in tqdm.tqdm(data, desc=desc):
            task = task_re.findall(ex['prompt'])[0].rstrip('. ')
            before = before_re.findall(ex['prompt'])[0].rstrip('. ')
            feedback = defaultdict(list)
            for step_str, reason in feedback_re.findall(ex['response']):
                for step in feedback_step_re.findall(step_str):
                    nps = True
                    if force_nps:
                        doc = nlp(reason)
                        nps = []
                        for x in doc.noun_chunks:
                            if any(skip in x.text.lower() for skip in ['where', 'which', 'player', 'left', 'right', 'straight', 'forward', 'goal', 'turn', 'them', 'they', 'street', 'direction', 'him', 'her', 'she', 'task', 'traffic', 'intersection', 'instruction', 'half', 'orientation', 'road', 'cordance', 'step', 'perspective', 'environment', 'this', 'path', 'line', 'car', 'choice', 'their', 'consisten', 'track', 'building', 'it', 'inormation']):
                                continue
                            nps.append(x)
                    if nps:
                        feedback[step].append(reason)
                        # print(reason)
                        # print(nps)
                        # import pdb; pdb.set_trace()
                    # if len(feedback[step]) > 1:
                    #     f = feedback[step]
                    #     import pdb; pdb.set_trace()
            steps = []
            for step_id, action, after in step_re.findall(ex['prompt']):
                steps.append(dict(
                    task=task,
                    step=step_id,
                    action=action,
                    before=before,
                    after=after,
                    label=step_id in feedback,
                    feedback=feedback[step_id],
                ))
                before = after

            # for s in steps:
            #     if s['label']:
            #         print(s)
            # import pdb; pdb.set_trace()
            proc.extend(steps)
        if p_pos:
            pos = [x for x in proc if x['label']]
            neg = [x for x in proc if not x['label']]
            random.Random(0).shuffle(neg)
            cat = pos + neg[:int(len(pos) / p_pos) - len(pos)]
            random.Random(0).shuffle(cat)
            proc = cat
        labels = Counter([x['label'] for x in proc])
        dataset = cls(proc)
        print(labels)
        return dataset

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_input(cls, r): return 'Task: {}\nBefore: {}\nAction: {}\nAfter: {}\nQuestion: was this helpful?\nAnswer:'.format(r['task'], r['before'], r['action'], r['after'])

    @classmethod
    def get_output(cls, label, reason=None):
        if reason is None:
            return label
        else:
            # return label
            return '{}. {}'.format(label, reason)

    def __getitem__(self, index):
        r = self.data[index]
        inp = self.get_input(r)
        out = self.get_output('Yes' if r['label'] else 'No', reason=' '.join(r['feedback']) if self.give_reason else None)
        # print(inp)
        # print()
        # print(out)
        # import pdb; pdb.set_trace()
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [inp], max_length=self.max_len_input,
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [out], max_length=self.max_len_output,
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        )

        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()
        src_mask = tokenized_inputs["attention_mask"].squeeze()
        target_mask = tokenized_targets["attention_mask"].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels, "label_boolean": r['label']}

    @classmethod
    def convert_trajectories(cls, ddata, tokenizer, limit=None, **kwargs):
        d = []
        fnames = list(glob.glob(ddata.joinpath('*.bz2').as_posix()))
        for fname in fnames:
            try:
                with bz2.open(fname) as f:
                    data = json.load(f)
            except Exception:
                print('cannot read {}'.format(fname))
                continue
            before = task = None
            for step, x in enumerate(data['traj']):
                if step == 0:
                    task = x['after']['obs']['instruction']
                    before = x['after']['obs']['observation']
                    continue
                assert before is not None
                assert task is not None
                after = x['after']['obs']['observation']
                action = x['action']
                if isinstance(action, dict):
                    action = action['key']
                y = dict(
                    task=task,
                    before=before,
                    after=after,
                    action=action,
                    label=False,
                    feedback='',
                    fname=fname,
                    step=step,
                    orig=x,
                )
                before = after
                d.append(y)
                if limit and len(d) >= limit:
                    break
        dataset = cls(d, tokenizer=tokenizer)
        return dataset, d


def convert(
    inp_dir: Path,
    out_dir: Path,
    seed: int = 37,
    ptrain: float = 0.8,
    p_pos: float = 0,
    p_pos_val: float = None,
    force_nps: bool = False,
):
    data = []
    for fname in inp_dir.glob('*.json.bz2'):
        with bz2.open(fname, 'rt') as f:
            data.append(json.load(f))

    if p_pos_val is None:
        p_pos_val = p_pos

    random.seed(seed)
    random.shuffle(data)

    out_dir = out_dir.joinpath(str(p_pos))
    if not out_dir.exists():
        os.makedirs(out_dir.as_posix())

    ntrain = int(ptrain * len(data))
    train = FeedbackDataset.make(data[:ntrain], p_pos=p_pos, force_nps=force_nps)
    train.save(out_dir.joinpath('train.json.bz2'), inp_dir=inp_dir.as_posix(), seed=seed, ptrain=ptrain)

    if ntrain < len(data):
        val = FeedbackDataset.make(data[ntrain:], p_pos=p_pos_val, force_nps=force_nps)
        val.save(out_dir.joinpath('val.json.bz2'), inp_dir=inp_dir.as_posix(), seed=seed, ptrain=ptrain)


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(convert)
