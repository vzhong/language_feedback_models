import unittest
from tqdm import auto as tqdm
from verbenvs.alfworld import VerbalizedALFWorld


class TestALFWorld(unittest.TestCase):

    @classmethod
    def make_env(cls, split):
        return VerbalizedALFWorld(split=split)

    def setUp(self):
        self.env = self.make_env('eval_out_of_distribution')

    def tearDown(self):
        self.env.close()

    def test_total(self):
        self.assertEqual(self.env.num_settings, 134)

    def test_runs(self):
        obs, info = self.env.reset()
        actions = info['admissible_actions']
        self.env.step(actions[0])


if __name__ == '__main__':
    unittest.main()
