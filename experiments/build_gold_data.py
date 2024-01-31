import os
import bz2
import copy
import tqdm
import json
import hydra
import torch
from prompt_utils import get_prompt
from verbenvs.auto import load_verbenv_by_name


@hydra.main(config_path='conf', config_name='gold', version_base='1.1')
def main(config):
    VEnv = load_verbenv_by_name(config.env.name)
    kwargs = {}
    if 'touchdown' in config.env.name:
        kwargs['device'] = torch.device('cuda')
    for split in config.env.splits.split(','):
        num_won = num_total = 0
        env = VEnv(deterministic=True, max_steps=config.env.max_steps, split=split, invalid_action=config.env.invalid_action, **kwargs)

        for _ in (pbar := tqdm.trange(env.num_settings, desc='setting for {}'.format(split))):
            try:
                obs, info = env.reset()
            except Exception as e:
                print(e)
                continue
            setting_id = env.curr_setting_id
            fout = '{}-{}.json.bz2'.format(split, setting_id)
            if os.path.isfile(fout):
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
                prompt=None,
            ))
            iterator = tqdm.trange if config.logging.verbose_step else range
            won = False

            gold_trajectory = getattr(env, 'gold_trajectory', None)
            for t in iterator(config.env.max_steps if gold_trajectory is None else len(gold_trajectory)):
                prompt = get_prompt(traj, config.prompt, action_to_string=env.extract_string_from_action, info=info)
                if gold_trajectory:
                    gold_action = gold_trajectory[t]
                else:
                    gold_action = env.get_expert_action()
                txt = env.extract_string_from_action(gold_action)
                obs, reward, terminated, info = env.step(txt)
                traj.append(dict(
                    after=dict(
                        obs=copy.deepcopy(obs),
                        info=copy.deepcopy(info),
                        reward=reward,
                        terminated=terminated,
                    ),
                    action=env.extract_string_from_action(gold_action),
                    prompt=prompt,
                ))

                if terminated:
                    won = reward > 0
                    break

            num_won += won
            num_total += 1
            pbar.set_description('{}/{}'.format(num_won, num_total))

            if not won:
                print('Did not win by following gold trajectory for {} after {} steps'.format(setting_id, len(traj)))
            with bz2.open(fout, 'wt') as f:
                json.dump(dict(
                    traj=traj,
                    won=won,
                    setting_id=setting_id,
                ), f, indent=2)
        env.close()

    print('done!')


if __name__ == '__main__':
    main()
