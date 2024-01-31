import os
import bz2
import tqdm
import json
import openai
import yaml
import random
import logging
import tenacity as T
from jsonargparse import CLI


def setup_openai(fname, mode='openai'):
    assert mode in {'openai'}
    with open(fname) as f:
        secrets = json.load(f)
    openai.api_version = "2023-03-15"
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = secrets['OPENAI_KEY']
    return secrets


@T.retry(stop=T.stop_after_attempt(20), wait=T.wait_fixed(10), after=lambda s: logging.error(repr(s)))
def query_openai(prompt, mode='openai', model='gpt-35-turbo', **kwargs):
    if mode == 'openai':
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs
        )
    else:
        response = openai.ChatCompletion.create(
            deployment_id=model,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs,
        )
    return response


STEP = """
----------
Step {step}
{render}
""".strip()


def adjust_render(render, truncate=7500):
    render = render.strip()
    step_start = render.index('Observation:')
    head = '\n'.join([line for line in render[:step_start].strip().splitlines() if not line.startswith('Step')])
    render = render[step_start:].strip()
    cache = []
    steps = []
    for line in render.splitlines()[1:]:  # take of the step count
        line = line.strip()
        if not line:
            continue
        if line.startswith('Available actions'):
            cache.clear()  # abandon this turn because it has not been completed
            break
        cache.append(line)
        if line.startswith('You got a reward'):
            steps.append('\n'.join(cache))
            cache.clear()
    if cache:
        steps.append('\n'.join(cache))
        cache.clear()
    render = []
    for i, step in enumerate(steps):
        render.append(STEP.format(step=i, render=step))
    render = '\n'.join(render)
    if len(render) > truncate:
        render = '...' + render[-truncate:]
    return head + '\n' + render


PROMPT = """
You will be shown a playthrough for solving a task. Afterwards, answer some questions about how you performed.
{render}
----------
Did you see anything that makes you believe you are on the right track to solve the task? Answer yes or no. If yes, then list the steps when you saw helpul things by the step number and explain why these steps were helpful. List only in bullet form.
""".strip()


PROMPT_INTENT = """
You will be shown a playthrough for solving a task.
{render}
----------
Did you see anything that makes you believe the player is on the right track to solve the task? Answer yes or no. If yes, then list the steps when you saw helpul things by the step number and explain why these steps were helpful. List only in bullet form. Only list important helpful steps. Next under heading #Summary, summarize in one sentence what the player doing at a high level. Finally under heading #Improvement, describe how the player can improve their strategy to solve the task.
""".strip()


PROMPT_REASON = """
You will be shown a playthrough for solving a task.
{render}
----------
For each step, list its step number, then answer "yes" or "no" to indicate whether it was helpful in solving the task, then finally give a reason on why. List only in bullet form.
""".strip()


PROMPT_ACTION = """
You will be shown a playthrough for solving a task.
{render}
----------
For steps where you think the player chose the wrong action, list its step number, then write the correct action to solve the task. Do not comment on the player action, just write what you think the correct action is. List only in bullet form.
""".strip()


def get_snapshots(data, snapshot_len=20):
    snapshots = []
    for ins in data:
        traj = ins['data']['traj']
        task = traj[0]['after']['obs']['instruction']
        for start in range(1, len(traj), snapshot_len):
            window = traj[start:start+snapshot_len]
            step_tm1 = traj[start-1]
            replay = ['Task: {}\nBefore: {}'.format(task, step_tm1['after']['obs']['observation'])]
            for i, step in enumerate(window):
                replay.append('-' * 10)
                replay.append('Step {}'.format(start+i))
                replay.append('Your action: {}'.format(step['action']))
                replay.append('Result: {}'.format(step['after']['obs']['observation']).strip())
                # replay.append('Reward: {}'.format(step['after']['reward']))
                step_tm1 = step
            r = '\n'.join(replay)
            snapshots.append(dict(
                fname=ins['fname'],
                render=r,
                render_start=start,
                render_end=start+len(window)-1,
            ))
    return snapshots


def query_llm(dtraj: str, fconfig: str = 'conf/feedback.yaml', dry_run=True, seed=37, limit=10000, feedback_style='intent', regenerate=False):
    with open(fconfig) as f:
        config = yaml.safe_load(f)

    print('Loading API Endpoint')
    setup_openai(config['api']['fsecrets'], mode=config['api']['mode'])

    print('Loading data')
    data = []
    for i, fname in enumerate(tqdm.tqdm(os.listdir(dtraj))):
        if fname.endswith('.json.bz2'):
            fin = os.path.join(dtraj, fname)
            try:
                with bz2.open(fin, 'rt') as f:
                    d = json.load(f)
                    data.append(dict(
                        fname=fin,
                        data=d,
                        success=any(x['after']['reward'] for x in reversed(d['traj'])),
                    ))
            except Exception:
                print('Could not open {}'.format(fin))

    print('Getting snapshots from {} trajectories'.format(len(data)))
    success = get_snapshots([x for x in data if x['success']])
    failure = get_snapshots([x for x in data if not x['success']])

    random.seed(seed)
    random.shuffle(success)
    random.shuffle(failure)
    snapshots = success[:limit//2]
    snapshots += failure[:limit - len(snapshots)]
    random.shuffle(snapshots)

    print('Querying API for {} examples'.format(len(snapshots)))
    dout = os.path.join(dtraj, 'feedback', feedback_style)
    if not os.path.isdir(dout):
        os.makedirs(dout)
    P = {
        'positive': PROMPT,
        'intent': PROMPT_INTENT,
        'reason': PROMPT_REASON,
        'action': PROMPT_ACTION,
    }[feedback_style]
    for i, ex in tqdm.tqdm(enumerate(snapshots), total=len(snapshots)):
        # prompt = P.format(render=adjust_render(ex['render']))
        prompt = P.format(render=ex['render'])
        entry = dict(prompt=prompt, fname=ex['fname'], response="", snapshot=ex)
        fout = os.path.join(dout, 'feedback.{}.json.bz2'.format(i))
        if not dry_run and ((not os.path.isfile(fout)) or regenerate):
            try:
                response = query_openai(prompt, max_tokens=config['api']['max_tokens'], model=config['api']['model'])
                entry['response'] = response['choices'][0]['message']['content']
                # print(entry['response'])
                # import pdb; pdb.set_trace()
            except Exception as e:
                print('Error in {}'.format(fout))
                print(e)
                continue
            else:
                assert fout != ex['fname']
                with bz2.open(fout, 'wt') as f:
                    json.dump(entry, f, indent=2)
    print('done!')


if __name__ == "__main__":
    CLI(query_llm)
