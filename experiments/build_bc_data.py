import os
import bz2
import copy
import glob
import ujson as json
from tqdm import auto as tqdm
from pathlib import Path
from torch.utils.data import Dataset


class BCDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_len_input=None, max_len_output=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

    def save(self, fout):
        with bz2.open(fout, 'wt') as f:
            json.dump(self.data, f, indent=2)

    @classmethod
    def load(cls, fout, **kwargs):
        print('loading data from {}'.format(fout))
        with bz2.open(fout, 'rt') as f:
            data = json.load(f)
        return cls(data, **kwargs)

    @classmethod
    def load_many(cls, fouts, **kwargs):
        data = []
        seen = set()
        for fout in fouts:
            print('loading data from {}'.format(fout))
            with bz2.open(fout, 'rt') as f:
                x = json.load(f)
            orig = len(seen)
            for xi in x:
                key = (xi['inp'], xi['out'])
                if key not in seen:
                    seen.add(key)
                    data.append(xi)
            print('added {} out of {}'.format(len(seen) - orig, len(x)))
        return cls(data, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # tokenize inputs
        r = self.data[index]
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [r['inp']], max_length=self.max_len_input,
            truncation=True,
            return_tensors="pt",
            padding='max_length',
        )
        # tokenize targets
        out = r['out']
        if isinstance(out, dict):
            out = out['key']  # hack for touchdown
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
                "labels": labels}


def iter_context(traj, context_window, label=True):
    total_reward = 0
    stitch = []
    for t in range(len(traj)-1):
        stm1 = traj[t]
        st = traj[t+1]
        stitch.append(dict(
            before=stm1['after']['obs']['observation'],
            after=st['after']['obs']['observation'],
            ins=stm1['after']['obs']['instruction'].replace('..', '.'),
            action=st['action'],
            reward=st['after']['reward'],
            time=t,
        ))

    assert len(stitch) == len(traj) - 1  # the first time step is a fake one

    if not label:  # add the last timestep for inference
        stm1 = traj[-1]
        stitch.append(dict(
            before=stm1['after']['obs']['observation'],
            after='',
            ins=stm1['after']['obs']['instruction'].replace('..', '.'),
            action=None,
            reward=0,
            time=len(traj)+1,
        ))

    for i in range(len(stitch)):
        total_reward += stitch[i]['reward']
        context = stitch[max(0, i-context_window):i+1]
        yield context, total_reward


PROMPT = """
Task: {task};
Score: {score};
Observation: {observation};
Action history:
{history};
""".strip()
PROMPT_STEP = """
[
Time: t-{time}
Observation: {observation};
Action: {action};
Reward: {reward};
""".strip()

PROMPT_ORDERED_HEAD = """
Task: {task};
Score: {score};
""".strip()
PROMPT_ORDERED_TAIL = """
[
Time: t
Observation: {observation};
Action: 
""".lstrip().rstrip('\n')


def build_example(context, total_reward, reverse=True):
    # The input format is as follows: “Task: D; Time: t − 1; Score: St−1; Action history: [At−i (+Rt−i) → Ot−i ] /* i loops from K to 1*/; Current room: Et−1; Inventory: It−1; Visited rooms: {E∗ 1, . . . , E∗ t−1}”
    context = copy.deepcopy(context)
    scienceworld_markers = ['This room is called the', 'This outside location is called the']
    is_scienceworld = any(x in context[0]['before'] for x in scienceworld_markers)

    if is_scienceworld:
        # from swiftsage paper
        for c in context:
            for x in scienceworld_markers:
                if c['before'].count(x) == 2:
                    c['before'] = c['before'][c['before'].index(x, 1):]

    last = context[-1]
    history = []

    for t, step in enumerate(context[:-1]):
        o = step['before'].strip()
        if is_scienceworld:
            # from swiftsage paper
            for x in scienceworld_markers:
                if x in o:
                    o = o[:o.index(x)]
        history.append(PROMPT_STEP.format(
            time=len(context)-t-1,
            action=step['action'],
            reward=step['reward'],
            observation=o,
        ))
    if reverse:
        history = list(reversed(history))
        inp = PROMPT.format(
            task=last['ins'],
            score=total_reward,
            observation=last['before'].strip(),
            history='\n'.join(history)
        )
    else:
        inp = PROMPT_ORDERED_HEAD.format(
            task=last['ins'],
            score=total_reward,
        )
        if history:
            inp += '\n' + '\n'.join(history)
        inp += '\n' + PROMPT_ORDERED_TAIL.format(observation=last['before'].strip())
    out = last['action']

    if is_scienceworld:
        # from swiftsage paper
        inp = inp.replace('(containing nothing)', '')
        inp = inp.replace('(that is open)', '')
        inp = inp.replace('green house', 'greenhouse')
        if out is not None:
            out = out.replace('green house', 'greenhouse')
        inp = inp.replace('\n\t', ' | ')
        inp = inp.replace('\n\t', ' | ')
    # print(inp)
    # print(out)
    # import pdb; pdb.set_trace()
    return dict(inp=inp, out=out)


def convert_files(fnames, context_window, fout, reverse=True):
    if not fout.parent.exists():
        os.makedirs(fout.parent.as_posix())
    dataset = []
    for fname in tqdm.tqdm(list(fnames)):
        with bz2.open(fname, 'rt') as f:
            traj = json.load(f)['traj']
            for context, total_reward in iter_context(traj, context_window):
                example = build_example(context, total_reward, reverse=reverse)
                dataset.append(example)
    d = BCDataset(dataset)
    d.save(fout)


def convert(
    out_dir: Path = Path("data/bc"),
    context_window: int = 20,
    env: str = 'alfworld',
    regenerate: bool = False,
):
    for split in ['train', 'eval']:
        fout = out_dir.joinpath(env, 'context={}'.format(context_window), '{}.json.bz2'.format(split))
        if split == 'eval':
            split_prefix = dict(
                alfworld='eval',
                scienceworld='dev',
                sparse_scienceworld='dev',
                touchdown='dev',
                descriptive_touchdown='dev',
            )[env]
        else:
            split_prefix = split
        if not fout.exists() or regenerate:
            print('making {}'.format(fout))
            convert_files(
                glob.glob('data/gold/{}/response/{}*.json.bz2'.format(env, split_prefix)),
                context_window,
                fout,
            )


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(convert)
