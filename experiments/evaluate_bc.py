import os
import bz2
import copy
import tqdm
import json
import torch
import hydra
import random
from pathlib import Path
from verbenvs.auto import load_verbenv_by_name
from build_bc_data import build_example, iter_context
from train_bc import LMAgent
from hydra.utils import to_absolute_path


def evaluate_model(cfg, model, dsave, random_policy=False):
    if not dsave.exists():
        os.makedirs(dsave.as_posix())

    random.seed(cfg.seed)
    print('loading environment')
    VEnv = load_verbenv_by_name(cfg.env)
    split = cfg.eval.split
    if split == 'dev' and cfg.env == 'alfworld':
        split = 'eval_out_of_distribution'
    env = VEnv(
       deterministic=True,
       max_steps=cfg.eval.max_steps,
       split=split,
       invalid_action='fuzzy',
    )

    max_settings = env.num_settings
    bar = tqdm.trange(max_settings, desc='setting')
    outcomes = []
    for bar_i in bar:
        try:
            obs, info = env.reset()
        except RuntimeError as e:
            # some envs cannot be verbalized
            print(e)
            continue
        setting_id = env.curr_setting_id
        fout = dsave.joinpath('{}.json.bz2'.format(setting_id))
        if os.path.isfile(fout):
            continue
        if cfg.eval.num_shards and bar_i % cfg.eval.num_shards != cfg.eval.shard:
            continue

        if cfg.eval.subsample > 0:
            if random.uniform(0, 1) > cfg.eval.subsample:
                continue

        traj = []
        traj.append(dict(
            after=dict(
                obs=copy.deepcopy(obs),
                info=copy.deepcopy(info),
                reward=None,
                terminated=None,
            ),
            action=None,
        ))
        iterator = range
        won = False
        for t in iterator(cfg.eval.max_steps):
            context, total_reward = list(iter_context(traj, context_window=cfg.context_window, label=False))[-1]
            example = build_example(context, total_reward, cfg.env)
            example['admissible_actions'] = info['admissible_actions'][:]
            if random_policy:
                action = random.choice(example['admissible_actions'])
                action_scores = None
            else:
                if cfg.eval.aggregation == 'generate':
                    action = model.generate_action(example, num_beams=cfg.eval.num_beams)
                    action_scores = None
                else:
                    action, action_scores = model.choose_action(example, aggregation=cfg.eval.aggregation, sample=cfg.eval.sample, eval_batch_size=cfg.eval.batch_size)

            # print(example['inp'])
            # print()
            # print(action)
            # import pdb; pdb.set_trace()

            obs, reward, terminated, info = env.step(action)
            traj.append(dict(
                after=dict(
                    obs=copy.deepcopy(obs),
                    info=copy.deepcopy(info),
                    reward=reward,
                    terminated=terminated,
                ),
                action=action,
                action_scores=action_scores,
                prompt=example['inp'],
            ))

            st = traj[-1]
            stm1 = traj[-2]
            context.append(dict(
                before=stm1['after']['obs']['observation'],
                after=st['after']['obs']['observation'],
                ins=st['after']['obs']['instruction'].replace('..', '.'),
                action=st['action'],
                reward=st['after']['reward'],
                time=t,
            ))

            if terminated:
                if reward > 0:
                    won = True
                break

        outcomes.append(won)
        bar.set_description('{}/{}'.format(sum(outcomes), len(outcomes)))

        with bz2.open(fout, 'wt') as f:
            json.dump(dict(
                traj=traj,
                setting_id=setting_id,
            ), f, indent=2)

    bar.close()
    print('done!')
    with open(dsave.joinpath('outcomes.json'), 'wt') as f:
        json.dump(outcomes, f)


@hydra.main(config_path='conf', config_name='bc', version_base='1.1')
def main(cfg):
    if cfg.eval.dsave:
        dsave = cfg.eval.dsave
    else:
        dsave = os.getcwd()
    saves = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.ckpt')]
    if len(saves) > 1:
        raise NotImplementedError('Cannot decide which save to use: {}'.format(saves))
    fsave = Path(os.getcwd(), saves[0])

    if cfg.eval.dout:
        dout = Path(cfg.eval.dout)
    else:
        dout = Path(os.getcwd(), 'eval', cfg.eval.split, 'aggregation={}'.format(cfg.eval.aggregation))
    print('loading model from {}'.format(fsave.absolute()))
    model = LMAgent.load_from_checkpoint(to_absolute_path(fsave), model=cfg.model)
    model.eval()
    evaluate_model(cfg, model, dout)


if __name__ == '__main__':
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass
    main()
