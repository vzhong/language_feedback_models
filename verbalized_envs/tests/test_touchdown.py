import unittest
from tqdm import auto as tqdm
from verbenvs.touchdown import VerbalizedTouchdown


class TestTouchdown(unittest.TestCase):

    @classmethod
    def make_env(cls, split):
        return VerbalizedTouchdown(split=split)

    def setUp(self):
        self.env = self.make_env('dev')

    def tearDown(self):
        self.env.close()

    def test_total(self):
        self.assertEqual(self.env.num_settings, 1391)

    def test_runs(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.curr_setting_id, 'dev-0')
        actions = info['admissible_actions']
        self.env.step(self.env.extract_string_from_action(actions[0]))


if __name__ == '__main__':
    unittest.main()