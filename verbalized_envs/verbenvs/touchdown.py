import os
import json
import torch
import networkx as nx
from pathlib import Path
from .base import VerbalizedEnv
from collections import defaultdict
from .clip_annotator import CLIPAnnotator


def turn_to_direction(turn):
    step = 360/16
    if turn >= step*15 or turn < step:
        return 'straight ahead'
    elif turn < step*3:
        return 'slightly to your right'
    elif turn < step*5:
        return 'to your right'
    elif turn < step*7:
        return 'behind you, slightly to your right'
    elif turn < step*9:
        return 'behind you'
    elif turn < step*11:
        return 'behind you, sightly to your left'
    elif turn < step*13:
        return 'to your left'
    elif turn < step*15:
        return 'slightly to your left'
    else:
        raise NotImplementedError()


class VerbalizedTouchdown(VerbalizedEnv):

    def __init__(self, dsplits: Path = None, fnps: Path = None, dgraph: Path = None, dpano_emb: Path = None, clip_cutoff=0.2, device=torch.device('cpu'), np_mode='instruction', max_instruction_len=512, max_observation_len=4000, max_action_len=20, verbalize_ylim=(300, 1200), turn_to_lang=turn_to_direction, filter_nps=('front',), **kwargs):
        self.dsplits = dsplits or self.infer_path().joinpath('touchdown')
        self.dpano_emb = dpano_emb or self.infer_path().joinpath('touchdown', 'panos')
        self.dgraph = dgraph or self.infer_path().joinpath('touchdown', 'graph')
        self.fnps = fnps or self.infer_path().joinpath('touchdown', 'nps.pt')

        self.clip_cutoff = clip_cutoff
        self.device = device
        self.np_mode = np_mode
        self.verbalize_ylim = verbalize_ylim
        self.turn_to_lang = turn_to_lang
        self.filter_nps = set(filter_nps)
        self.graph = self.load_graph(self.dgraph)
        assert np_mode in {'instruction', 'all'}

        self.nps, self.nps_emb = CLIPAnnotator.load_nps(self.fnps)

        # states
        self.trajs = None
        self.curr_traj_i = 0
        self.curr_pano = None
        self.curr_heading = None
        self.curr_scan = None

        self._gold_traj = None

        super().__init__(
            max_instruction_len=max_instruction_len,
            max_observation_len=max_observation_len,
            max_action_len=max_action_len,
            **kwargs,
        )

    @property
    def curr_setting_id(self):
        return '{}-{}'.format(self.split, self.curr_traj_i)

    @property
    def num_settings(self):
        return len(self.trajs)

    def set_split(self, split):
        with open(os.path.join(self.dsplits, '{}.json'.format(split))) as f:
            self.trajs = json.load(f)

    @property
    def curr_traj(self):
        return self.trajs[self.curr_traj_i]

    @property
    def curr_nps(self):
        if self.np_mode == 'instruction':
            return self.curr_traj['noun_chunks']
        else:
            return self.nps

    @property
    def curr_target_pano(self):
        return self.curr_traj['route_panoids'][-1]

    def set_traj(self, i):
        self.curr_traj_i = i
        self.curr_pano = self.curr_traj['route_panoids'][0]
        self.curr_heading = self.curr_traj['start_heading']
        self.curr_instruction = self.curr_traj['navigation_text']
        self._gold_traj = None

    def step(self, action):
        if isinstance(action, dict):
            action = action['key']
        return super().step(action)

    def reset_to_random(self):
        self.set_traj(self.np_random.integers(0, len(self.trajs)))
        curr_verb, curr_actions = self.verbalize()
        return curr_verb, curr_actions

    def reset_to_next(self):
        self.set_traj(0 if self.curr_pano is None else (self.curr_traj_i + 1) % len(self.trajs))
        curr_verb, curr_actions = self.verbalize()
        return curr_verb, curr_actions

    def extract_string_from_action(self, action):
        return action['key']

    def take_action(self, action):
        # transition environment and return obs, reward, terminated
        self.curr_pano = action['new_pano']
        self.curr_heading = action['new_heading']
        terminated = self.curr_pano == self.curr_target_pano
        reward = 1 if terminated else 0  # Binary sparse rewards
        curr_verb, curr_actions = self.verbalize()
        return curr_verb, reward, terminated, curr_actions

    @classmethod
    def load_graph(cls, dgraph):
        g = nx.DiGraph()
        with open(os.path.join(dgraph, 'nodes.txt')) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                g.add_node(panoid, yaw_angle=int(pano_yaw_angle), lat=float(lat), lng=float(lng))
        with open(os.path.join(dgraph, 'links.txt')) as f:
            for line in f:
                start_panoid, heading, end_panoid = line.strip().split(',')
                g.add_edge(start_panoid, end_panoid, heading=float(heading))
        return g

    @classmethod
    def heading_to_x(cls, heading, yaw, xmax=3000):
        centre = xmax // 2
        shift_angle = yaw - heading
        shift = int(xmax * shift_angle / 360)
        heading_x = (centre - shift) % xmax
        return heading_x

    def show_image(self, dpano, figsize=(20, 10), fontsize=14, annotate_prob=False):
        from matplotlib import pyplot as plt
        from PIL import Image
        import numpy as np
        pano = self.curr_pano
        heading = self.curr_heading
        cutoff = self.clip_cutoff

        scan = CLIPAnnotator.scan(
            self.dpano_emb,
            self.curr_nps,
            pano,
            self.nps,
            self.nps_emb,
            self.device,
        )
        yaw = self.graph.nodes[pano]['yaw_angle']
        im = np.asarray(Image.open(os.path.join(dpano, pano + '.jpg')))
        xmax = im.shape[1]
        ymin, ymax = self.verbalize_ylim
        curr_x = self.heading_to_x(heading, yaw, xmax)
        plt.figure(figsize=figsize)
        plt.imshow(im, interpolation='nearest', aspect='auto')
        for a in scan:
            x, y = a['j'] + a['step_size']//2, a['i'] + a['step_size']//2
            # turn = ((x - curr_x) / xmax * 360) % 360
            if ymin < y < ymax:
                top = a['scores'][0]
                if cutoff:
                    if top[1] < cutoff:
                        continue
                s = top[0]
                if annotate_prob:
                    s += ' {}'.format(top[1])
                plt.annotate(s, xy=(x, y), xytext=(x, y), arrowprops=dict(facecolor='red'), fontsize=fontsize, rotation=60)

        # annotate headings
        xy = self.heading_to_x(heading, yaw, xmax=xmax), 0
        plt.annotate('Current heading', xy=xy, xytext=xy, arrowprops=dict(facecolor='blue'), fontsize=fontsize, rotation=60)

        for dest in self.graph.neighbors(pano):
            edge = self.graph.edges[pano, dest]
            dest_xy = self.heading_to_x(edge['heading'], yaw, xmax=xmax), 0
            plt.annotate('GO to {}'.format(dest), xy=dest_xy, xytext=dest_xy, arrowprops=dict(facecolor='green'), fontsize=fontsize, rotation=60)
        plt.show()

    def verbalize(self, xmax=3000):
        pano = self.curr_pano
        heading = self.curr_heading
        scan = CLIPAnnotator.scan(
            self.dpano_emb,
            self.curr_nps,
            pano,
            self.nps,
            self.nps_emb,
            self.device,
        )
        desc = []
        ymin, ymax = self.verbalize_ylim
        yaw = self.graph.nodes[pano]['yaw_angle']
        curr_x = self.heading_to_x(heading, yaw, xmax)
        mapped = {}
        landmarks = []
        for a in scan:
            x, y = a['j'] + a['step_size']//2, a['i'] + a['step_size']//2
            if ymin < y < ymax:
                top = a['scores'][0]
                if self.clip_cutoff:
                    if top[1] < self.clip_cutoff:
                        continue
                # a_heading = x / xmax * 360
                turn = ((x - curr_x) / xmax * 360) % 360
                key = self.turn_to_lang(turn)
                if key not in mapped:
                    mapped[key] = set()
                mapped[key].add(top[0])
                landmarks.append((turn, top[0]))
        for turn, items in mapped.items():
            items = list(items)
            if items:  # only verbalize directions with landmarks
                desc.append('{}, you see:'.format(turn))
                for i in items:
                    if i not in self.filter_nps:
                        desc.append('- {}'.format(i))
                desc.append('')

        actions = self.get_actions_for_pano(pano, xmax=xmax)
        resolved = self.resolve_collisions(actions, landmarks)
        return '\n'.join(desc), resolved

    def get_actions_for_pano(self, pano, xmax=3000):
        actions = []
        yaw = self.graph.nodes[pano]['yaw_angle']
        curr_x = self.heading_to_x(self.curr_heading, yaw, xmax)
        for dest in self.graph.neighbors(pano):
            edge = self.graph.edges[pano, dest]
            x = self.heading_to_x(edge['heading'], yaw, xmax=xmax)
            turn = ((x - curr_x) / xmax * 360) % 360
            actions.append(dict(turn=turn, key=self.turn_to_lang(turn), new_pano=dest, new_heading=edge['heading']))
        return actions

    @classmethod
    def resolve_collisions(cls, actions, landmarks):
        mapped_actions = defaultdict(list)
        for a in actions:
            mapped_actions[a['key']].append(a)

        resolved = []
        for key, vals in mapped_actions.items():
            if len(vals) > 1:
                # multiple actions got mapped to this key
                pano_to_landmark = cls.align_to_landmark(vals, landmarks)
                for a in vals:
                    r = a.copy()
                    r['key'] += ' towards {}'.format(pano_to_landmark[a['new_pano']])
                    resolved.append(r)
            else:
                resolved.extend(vals)

        if len(resolved) != len(set([a['key'] for a in resolved])):
            import pdb; pdb.set_trace()
            raise NotImplementedError('Found duplicate actions: {}'.format(resolved))

        return resolved

    @classmethod
    def align_to_landmark(cls, actions, landmarks):
        # map them relative to each landmarks
        aligned = []
        unmapped_landmarks = set()
        for a in actions:
            for turn, landmark in landmarks:
                aligned.append((a, landmark, abs(turn-a['turn'])))
                unmapped_landmarks.add(landmark)
        aligned.sort(key=lambda tup: tup[-1])
        mapped_panos = {}
        for action, landmark, delta in aligned:
            if action['new_pano'] in mapped_panos:
                continue
            if landmark in unmapped_landmarks:
                mapped_panos[action['new_pano']] = landmark
                unmapped_landmarks.remove(landmark)

        # map them relative to each other
        unmapped = defaultdict(list)
        for a in actions:
            if a['new_pano'] not in mapped_panos:
                unmapped[a['key']].append(a)
        for key, vals in unmapped.items():
            if len(vals) == 1:
                mapped_panos[vals[0]['new_pano']] = vals[0]['key']
            elif len(vals) == 2:
                a, b = vals
                if a['turn'] > b['turn']:
                    mapped_panos[a['new_pano']] = 'more to the right'
                    mapped_panos[b['new_pano']] = 'more to the left'
                else:
                    mapped_panos[a['new_pano']] = 'more to the left'
                    mapped_panos[b['new_pano']] = 'more to the right'
            else:
                raise Exception('got too many ({}): {}'.format(len(vals), vals))
        return mapped_panos

    @property
    def gold_traj(self):
        if self._gold_traj is None:
            gold_traj = self.curr_traj['route_panoids']
            if nx.is_path(self.graph, gold_traj):
                self._gold_traj = gold_traj
            else:
                self._gold_traj = nx.shortest_path(self.graph, self.curr_traj['route_panoids'][0], self.curr_target_pano)
        return self._gold_traj

    def get_expert_action(self):
        # look up gold traj
        if self.curr_pano not in self.gold_traj:
            import pdb; pdb.set_trace()
        i = self._gold_traj.index(self.curr_pano)
        nxt_pano = self._gold_traj[i+1]
        good_action = [a for a in self.curr_actions if a['new_pano'] == nxt_pano]
        if not good_action:
            import pdb; pdb.set_trace()
        if good_action[0]['new_pano'] not in self.gold_traj:
            import pdb; pdb.set_trace()
        return good_action[0]


class DescriptiveVerbalizedTouchdown(VerbalizedTouchdown):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, np_mode='all', **kwargs)
