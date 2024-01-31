from scienceworld import ScienceWorldEnv
from tqdm import auto as tqdm
from .base import VerbalizedEnv

import re


class VerbalizedScienceWorld(VerbalizedEnv):

    def __init__(self, simplifications=tuple(), max_instruction_len=256, max_observation_len=256, max_action_len=4000, invalid_action='force', enable_quit=False, **kwargs):
        self.get_variations = None
        self.simplifications = simplifications
        self.all_tasks = None
        self.orig_env = None
        self.enable_quit = enable_quit

        # states
        self.curr_task_i = None
        self.curr_variations = None
        self.curr_variation_i = None
        self.curr_action_types = None
        self.curr_look = None

        self._num_settings = None

        super().__init__(
            invalid_action=invalid_action,
            max_instruction_len=max_instruction_len,
            max_observation_len=max_observation_len,
            max_action_len=max_action_len,
            **kwargs,
        )

    def set_split(self, split):
        self.orig_env = ScienceWorldEnv(envStepLimit=self.max_steps)
        self.all_tasks = self.orig_env.getTaskNames()
        self.get_variations = dict(
            train=self.orig_env.getVariationsTrain,
            dev=self.orig_env.getVariationsDev,
            test=self.orig_env.getVariationsTest,
        )[split]

    @property
    def curr_setting_id(self):
        return '{}-task-{}_var-{}'.format(self.split, self.curr_task, self.curr_variation)

    @property
    def num_settings(self):
        if self._num_settings is None:
            total = 0
            for i in tqdm.trange(len(self.all_tasks), desc='Computing num_settings'):
                self.set_task(i)
                total += len(self.curr_variations)
            self._num_settings = total
        return self._num_settings

    @property
    def curr_task(self):
        return self.all_tasks[self.curr_task_i]

    @property
    def curr_variation(self):
        return self.curr_variations[self.curr_variation_i]

    @property
    def simplification_str(self):
        return ','.join(self.simplifications)

    def set_task(self, task_i):
        assert task_i < len(self.all_tasks)
        self.curr_task_i = task_i
        self.orig_env.load(self.curr_task, 0, self.simplification_str, generateGoldPath=True)
        self.curr_variations = self.get_variations()

    def set_variation(self, variation_i):
        assert variation_i < len(self.curr_variations)
        self.curr_variation_i = variation_i
        self.orig_env.load(self.curr_task, self.curr_variation, self.simplification_str, generateGoldPath=True)
        self.curr_instruction = self.orig_env.getTaskDescription() + ' Do not focus on objects not central to completing the task!'

    def filter_actions(self, actions):
        # from swiftsage
        rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]
        valid_open_door = ["open door to " + i for i in rooms]
        invalid_focus = ["focus on "+x for x in ["agent", "air"] + rooms]
        validActions = set(self.orig_env.getValidActionObjectCombinations())
        validActions.update(valid_open_door)
        validActions.difference_update(invalid_focus)

        inventory = self.orig_env.inventory().lower()

        validActions.difference_update(self.recent_actions[-3:])
        for va in list(validActions):
            if "door" in va and "open" not in va:
                validActions.remove(va)
                continue
            if va.startswith("focus on"):
                pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
                used_objs = pattern.sub("", va).split(" ")
                valid = True
                for obj in used_objs:
                    if obj not in self.curr_look + " " + inventory:
                        valid = False
                if not valid:
                    validActions.remove(va)
        return sorted(list(validActions))

    def reset_to_random(self):
        task_i = self.np_random.integers(0, len(self.all_tasks))
        self.set_task(task_i)
        variation_i = self.np_random.integers(0, len(self.curr_variations))
        self.set_variation(variation_i)

        obs, reward, terminated, orig_info = self.orig_env.step('look around')
        curr_actions = orig_info['valid']
        self.curr_look = orig_info['look']
        return self.verbalize(obs, orig_info), curr_actions

    def reset_to_next(self):
        if self.curr_variation_i is None:
            task_i = 0
            variation_i = 0
            self.set_task(task_i)
        else:
            # go to next variation
            task_i = self.curr_task_i
            variation_i = self.curr_variation_i + 1
            if variation_i >= len(self.curr_variations):
                # go to next task
                task_i = (task_i + 1) % len(self.all_tasks)
                variation_i = 0
                self.set_task(task_i)
        self.set_variation(variation_i)
        obs, reward, terminated, orig_info = self.orig_env.step('look around')
        self.curr_look = orig_info['look']
        curr_actions = orig_info['valid']
        return self.verbalize(obs, orig_info), curr_actions

    def extract_string_from_action(self, action):
        return action

    def take_action(self, action):
        # transition environment and return obs, reward, terminated
        obs, reward, terminated, orig_info = self.orig_env.step(action if isinstance(action, str) else self.extract_string_from_action(action))
        valid_actions = orig_info['valid']
        self.curr_look = orig_info['look']
        return self.verbalize(obs, orig_info), reward, terminated, valid_actions

    def verbalize(self, obs, orig_info):
        verb = obs + '\n' + orig_info['look'] + '\n' + orig_info['inv']
        return verb

    def _get_info(self):
        info = super()._get_info()
        info['possible_objects'] = self.orig_env.getPossibleObjects()
        info['possible_action_types'] = self.orig_env.getPossibleActions()
        return info

    @property
    def gold_trajectory(self):
        return self.orig_env.getGoldActionSequence()


class SparseVerbalizedScienceWorld(VerbalizedScienceWorld):

    def take_action(self, action):
        verb, reward, terminated, valid_actions = super().take_action(action)
        if not terminated:
            reward = 0
        return verb, reward, terminated, valid_actions
