import re
import random


def get_prompt(traj, config, action_to_string, info=None):
    last = traj[-1]
    prompt = ['You task is: {}'.format(last['after']['obs']['instruction']), '']

    for t in traj:
        if t['action'] is not None:
            prompt.append('You decide to: {}'.format(action_to_string(t['after']['info']['action_taken'])))
        if config.history == 'response':
            prompt.append('You see: {}'.format(t['after']['obs']['observation']))
            if t['after']['reward'] is not None:
                prompt.append('You got a reward of: {}'.format(t['after']['reward']))
        prompt.append('\n\n')

    if info is not None:
        phrases = []
        if 'possible_action_types' in info:
            phrases.append('You can take the following actions:')
            for o in info['possible_action_types']:
                phrases.append('- {}'.format(o))
            phrases.append('')
            phrases.append('the OBJ must be replaced by one of the following:')
            for o in info['possible_objects']:
                phrases.append('- {}'.format(o))
        prompt.append('\n'.join(phrases))
        prompt.append('')

    prompt.append('What do you decide to do?')
    choices = last['after']['info']['admissible_actions']
    if config.action.type == 'admissible':
        prompt.append('Available actions:')
        for i, a in enumerate(choices):
            if config.action.numbered:
                prompt.append('{}: {}'.format(i, action_to_string(a)))
            else:
                prompt.append('- {}'.format(a))
    elif config.action.type == 'examples':
        example_actions = choices[:config.action.num_examples]
        random.Random(0).shuffle(example_actions)
        prompt.append('For example, you can')
        for i, a in enumerate(example_actions):
            if config.action.numbered:
                prompt.append('{}: {}'.format(i, a))
            else:
                prompt.append('- {}'.format(a))
        prompt.append('...')
        prompt.append('There are more actions you can take.')
    else:
        assert config.action.type is None

    prompt.append('')
    prompt.append('You decide to: ')

    out = '\n'.join(prompt)
    return re.sub(r'\n\n+', '\n\n', out).strip()
