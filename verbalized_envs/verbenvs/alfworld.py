import os
import yaml
import textworld
import alfworld.agents.environment as E
from pathlib import Path
from .base import VerbalizedEnv
from textworld.gym import envs as tw_envs


class VerbalizedALFWorld(VerbalizedEnv):

    def __init__(self, dconfig: Path = None, ddata: Path = None, max_instruction_len=512, max_observation_len=4000, max_action_len=20, human_goal_prob=1.0, **kwargs):
        self.dconfig = dconfig or self.infer_path().joinpath('alfworld', 'config')
        self.ddata = ddata or self.infer_path().joinpath('alfworld', 'data')
        os.environ['ALFWORLD_DATA'] = self.ddata.absolute().as_posix()
        self.alfworld_conf = None
        self.orig_env = None
        self.tw_env = None
        self.human_goal_prob = human_goal_prob

        super().__init__(
            max_instruction_len=max_instruction_len,
            max_observation_len=max_observation_len,
            max_action_len=max_action_len,
            **kwargs,
        )

        self._curr_gamefile = None
        self._gold_action = None

    def close(self):
        self.tw_env.close()

    def set_split(self, split):
        self.alfworld_fconf = os.path.join(self.dconfig, 'alfred_config.yaml')
        assert os.path.exists(self.alfworld_fconf), "Invalid config file {}".format(self.alfworld_fconf)
        with open(self.alfworld_fconf) as reader:
            config = yaml.safe_load(reader)
        config['env']['goal_desc_human_anns_prob'] = 0 if split == 'train' else self.human_goal_prob
        env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
        self.orig_env = getattr(E, env_type)(config, train_eval=split)
        self.orig_env.seed(0)
        self.tw_env = tw_envs.TextworldGymEnv(
            self.orig_env.game_files,
            request_infos=textworld.EnvInfos(won=True, admissible_commands=True, expert_type=config["env"]["expert_type"], expert_plan=True, extras=["gamefile"]),
            max_episode_steps=self.max_steps,
            action_space=self.action_space,
            observation_space=self.observation_space,
            wrappers=[
                E.alfred_tw_env.AlfredDemangler(shuffle=not self.deterministic),
                E.alfred_tw_env.AlfredInfos,
            ],
        )

    @property
    def curr_setting_id(self):
        assert self._curr_gamefile is not None
        tags = os.path.normpath(self._curr_gamefile).split(os.path.sep)[-4:-1]
        return '__'.join(tags)

    @property
    def num_settings(self):
        return self.orig_env.num_games

    def reset_to_random(self):
        # randomization is handled by ALFWorld env in 'domain_randomization'
        obs, info = self.tw_env.reset()
        lines = obs.strip().split('\n')
        nontask = '\n'.join(lines[:-1]).split('\n\n')[-1]
        self.curr_instruction = lines[-1].replace('Your task is to: ', '')
        curr_verb, curr_actions = self.verbalize(nontask, info)
        self._curr_gamefile = info['extra.gamefile']
        self._gold_action = info['expert_plan'][0]
        return curr_verb, curr_actions

    def reset_to_next(self):
        # randomization is handled by ALFWorld env in 'domain_randomization'
        return self.reset_to_random()

    def extract_string_from_action(self, action):
        return action

    def take_action(self, action):
        # transition environment and return obs, reward, terminated
        obs, reward, terminated, info = self.tw_env.step(action)
        self._gold_action = info['expert_plan'][0]
        curr_verb, curr_actions = self.verbalize(obs, info)
        return curr_verb, reward, terminated, curr_actions

    def verbalize(self, obs, info):
        return obs, info['admissible_commands']

    def get_expert_action(self):
        return self._gold_action
