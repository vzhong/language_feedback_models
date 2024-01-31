import os
import gym
import editdistance
from pathlib import Path
from gym import spaces


class VerbalizedEnv(gym.Env):
    metadata = {"render_modes": ["human", "human_abbrev_actions", None]}

    @classmethod
    def infer_path(cls):
        if 'VERBENVS_DATA' in os.environ:
            root = Path(os.environ['VERBENVS_DATA'])
            if not root.is_dir():
                raise NotADirectoryError('Directory does not exist at {}!'.format(root))
        else:
            root = Path(os.path.abspath(__file__)).parent.parent.joinpath('data')
            if not root.is_dir():
                raise NotADirectoryError('Trying to infer $VERBENVS_DATA but environment variable has not been set! Please download data and set environment variable.')
        return root

    def __init__(self, render_mode=None, deterministic=True, split='train', max_steps=80, max_instruction_len=512, max_observation_len=1024, max_action_len=128, invalid_action='raise'):
        self.deterministic = deterministic
        self.split = split
        self.max_steps = max_steps
        self.observation_space = spaces.Dict(
            {
                "instruction": spaces.Text(max_length=max_instruction_len, min_length=1),
                "observation": spaces.Text(max_length=max_observation_len, min_length=1),
            }
        )
        self.action_space = spaces.Text(max_length=max_action_len, min_length=1)

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # states
        self.curr_actions = None
        self.curr_verb = None
        self.curr_instruction = None
        self.curr_steps = 0
        self.recent_actions = []
        self.invalid_action = invalid_action
        assert invalid_action in {'raise', 'fuzzy', 'force'}

        self.set_split(split)

    @property
    def num_settings(self):
        raise NotImplementedError()

    @property
    def curr_setting_id(self):
        raise NotImplementedError()

    def set_split(self):
        raise NotImplementedError()

    def reset_to_random(self):
        # return curr_verb, curr_actions
        raise NotImplementedError()

    def reset_to_next(self):
        # return curr_verb, curr_actions
        raise NotImplementedError()

    def extract_string_from_action(self, action):
        raise NotImplementedError()

    def take_action(self, action):
        # transition environment and return obs, reward, terminated, valid_actions
        raise NotImplementedError()

    def get_expert_action(self):
        # return expert action
        raise NotImplementedError()

    def filter_actions(self, actions):
        return actions

    def _get_obs(self):
        return {
            "instruction": self.curr_instruction,
            "observation": self.curr_verb,
        }

    def _get_info(self):
        return {
            'admissible_actions': self.curr_actions,
        }

    def reset(self, seed=None, options=None, filter_actions=True):
        super().reset(seed=seed)
        if self.deterministic:
            self.curr_verb, self.curr_actions = self.reset_to_next()
        else:
            self.curr_verb, self.curr_actions = self.reset_to_random()
        if filter_actions:
            self.curr_actions = self.filter_actions(self.curr_actions)
        if len(self.curr_actions) != len(set([self.extract_string_from_action(a) for a in self.curr_actions])):
            raise NotImplementedError('Found duplicate actions: {}'.format(self.current_actions))
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode and "human" in self.render_mode:
            self._render_frame()
        self.curr_steps = 0
        self.recent_actions.clear()
        return observation, info

    @classmethod
    def infer_action_index(cls, action, choices):
        matched = None
        if isinstance(action, int):
            matched = action
        elif isinstance(action, str):
            try:
                matched = int(action.strip().split()[0])
            except:
                pass
        if isinstance(matched, int):
            return matched if matched < len(choices) else None
        return None

    def step(self, action, filter_actions=True):
        choices = self.curr_actions
        if isinstance(action, str):
            action = action.strip()
            if not action:
                chosen = choices[0]
            else:
                action = action.split('\n')[0].strip()
                chosen = [a for a in choices if self.extract_string_from_action(a) == action]
                if chosen:
                    chosen = chosen[0]
                else:
                    if self.invalid_action == 'raise':
                        raise NotImplementedError('Cannot take action {}'.format(action))
                    elif self.invalid_action == 'fuzzy':
                        start = action[:self.action_space.max_length]
                        # match longest common subsequence
                        scores = []
                        for a in choices:
                            c = self.extract_string_from_action(a)
                            s = editdistance.eval(c, start)
                            scores.append((a, s))
                        scores.sort(key=lambda tup: tup[1])
                        chosen, score = scores[0]
                    elif self.invalid_action == 'force':
                        chosen = action
        else:
            raise NotImplementedError('Unsupported action {}'.format(action))

        self.curr_verb, reward, terminated, self.curr_actions = self.take_action(chosen)
        if filter_actions:
            self.curr_actions = self.filter_actions(self.curr_actions)
        observation = self._get_obs()
        info = self._get_info()
        info['action_taken'] = chosen
        if self.render_mode and "human" in self.render_mode:
            self._render_frame()
        self.curr_steps += 1
        if self.curr_steps > self.max_steps:
            terminated = True
            reward = self.get_terminal_reward(original_reward=reward)
        self.recent_actions.append(chosen)
        return observation, reward, terminated, info

    def get_terminal_reward(self, original_reward):
        return original_reward

    def _render_frame(self):
        print(self.curr_instruction)
        print()
        print(self.curr_verb)
        print()
        if self.render_mode == 'human_abbrev_actions':
            print(repr([self.extract_string_from_action(a) for a in self.curr_actions])[:100] + '...')
        else:
            print([self.extract_string_from_action(a) for a in self.curr_actions])

    def close(self):
        pass
