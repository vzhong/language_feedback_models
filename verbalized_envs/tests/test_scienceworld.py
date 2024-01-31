import unittest
from tqdm import auto as tqdm
from verbenvs.scienceworld import VerbalizedScienceWorld


class TestScienceWorld(unittest.TestCase):

    @classmethod
    def make_env(cls, split):
        return VerbalizedScienceWorld(split=split)

    def setUp(self):
        self.env = self.make_env('dev')

    def tearDown(self):
        self.env.close()

    def test_total(self):
        self.assertEqual(self.env.num_settings, 1796)

    def test_runs(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.curr_setting_id, 'dev-task-boil_var-14')
        actions = info['admissible_actions']
        self.env.step(actions[0])

if __name__ == '__main__':
    unittest.main()