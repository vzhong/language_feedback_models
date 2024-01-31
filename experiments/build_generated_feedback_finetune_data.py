import os
import bz2
import glob
import ujson as json
from tqdm import auto as tqdm
from pathlib import Path
from build_bc_data import BCDataset, iter_context, build_example


def convert_files(fnames, context_window, fout, strategy, reverse=True):
    if not fout.parent.exists():
        os.makedirs(fout.parent.as_posix())
    dataset = []
    kept = set()
    fnames = list(fnames)
    for fname in tqdm.tqdm(fnames):
        with bz2.open(fname, 'rt') as f:
            traj = json.load(f)
            if strategy == 'env_reward':
                success = [x['after']['reward'] for x in traj]
                if any(success):
                    for context, total_reward in iter_context(traj, context_window):
                        example = build_example(context, total_reward, reverse=reverse)
                        dataset.append(example)
                    kept.add(fname)
            elif strategy == 'pred_reward':
                # examples = []
                # for context, total_reward in iter_context(traj, context_window):
                #     examples.append(build_example(context, total_reward, reverse=reverse))
                # for x, ex in zip(traj[1:], examples):
                #     if x['llm_pred'].startswith('Yes'):
                #         kept.add(fname)
                #         dataset.append(ex)
                for ex in traj:
                    if ex['llm_pred'].startswith('Yes'):
                        kept.add(fname)
                        dataset.append(dict(inp=ex['prompt'], out=ex['action']))
                # import pdb; pdb.set_trace()
            elif strategy == 'pred_reward_traj':
                good = set()
                for i, ex in enumerate(traj):
                    if ex['llm_pred'].startswith('Yes'):
                        for i in range(max(0, i-5), i+1):
                            good.add(i)
                for i in sorted(list(good)):
                    ex = traj[i]
                    kept.add(fname)
                    dataset.append(dict(inp=ex['prompt'], out=ex['action']))
                # examples = []
                # for context, total_reward in iter_context(traj, context_window):
                #     examples.append(build_example(context, total_reward, reverse=reverse))
                # last_yes_t = None
                # t = 0
                # for x, ex in zip(traj[1:], examples):
                #     t += 1
                #     if x['llm_pred'].startswith('Yes'):
                #         last_yes_t = t
                # if last_yes_t is not None:
                #     kept.add(fname)
                #     for context, total_reward in iter_context(traj[:last_yes_t], context_window):
                #         example = build_example(context, total_reward, reverse=reverse)
                #         dataset.append(example)
            elif strategy == 'mixed_reward':
                success = [x['after']['reward'] for x in traj]
                if any(success):
                    for context, total_reward in iter_context(traj, context_window):
                        example = build_example(context, total_reward, reverse=reverse)
                        dataset.append(example)
                    kept.add(fname)
                else:
                    examples = []
                    for context, total_reward in iter_context(traj, context_window):
                        examples.append(build_example(context, total_reward, reverse=reverse))
                    last_yes_t = None
                    t = 0
                    for x, ex in zip(traj[1:], examples):
                        t += 1
                        if x['llm_pred'].startswith('Yes'):
                            last_yes_t = t
                    if last_yes_t is not None:
                        kept.add(fname)
                        for context, total_reward in iter_context(traj[:last_yes_t], context_window):
                            example = build_example(context, total_reward, reverse=reverse)
                            dataset.append(example)
            else:
                assert strategy == 'all'
                for context, total_reward in iter_context(traj, context_window):
                    example = build_example(context, total_reward, reverse=reverse)
                    dataset.append(example)
                kept.add(fname)
    print('kept {} traj out of {}'.format(len(kept), len(fnames)))
    d = BCDataset(dataset)
    d.save(fout)


def convert(
    inp_dir: Path = None,
    out_dir: Path = Path("data/bc"),
    context_window: int = 20,
    env: str = 'alfworld',
    strategy: str = 'all',
    regenerate: bool = False,
    p_pos: float = 0.0,
    expname: str = '',
):
    fout = out_dir.joinpath(env, 'context={}'.format(context_window), '{}gen-{}-{}.json.bz2'.format(expname, p_pos, strategy))
    if not fout.exists() or regenerate:
        print('making {}'.format(fout))
        convert_files(
            glob.glob(inp_dir.joinpath('*.bz2').as_posix()),
            context_window,
            fout,
            strategy=strategy,
        )


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(convert)
