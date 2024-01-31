import os
import bz2
import argparse
import pandas as pd
import ujson as json
from tqdm import auto as tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', default='outputs')
args = parser.parse_args()


data = []
for root, dirs, files in os.walk(args.root):
    if 'feedback' in root:
        continue
    if not files:
        continue
    for fname in tqdm.tqdm(files, desc=root):
        if fname.endswith('.json.bz2'):
            try:
                with bz2.open(os.path.join(root, fname)) as f:
                    entry = json.load(f)
                    traj = entry['traj']
            except Exception:
                print('skipping {}'.format(fname))
                continue
            won = False
            for step in traj:
                if step['after']['terminated'] and step['after']['reward'] > 0:
                    won = True
                    break
            data.append(dict(
                root=root,
                dataset='/'.join(root.split('/')[3:]),
                id=entry['setting_id'],
                len=len(traj),
                reward=sum([t['after']['reward'] for t in traj if t['after']['reward'] is not None]),
                terminated=traj[-1]['after']['terminated'],
                won=won,
            ))
df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
gb = df.groupby(by=['root'])

agg = dict(
    won=['mean', 'count'],
    # len=['min', 'max', 'mean'],
    # reward=['min', 'max', 'mean'],
    # terminated=['mean', 'count'],
)

res = gb.agg(agg)

print(res)
with open(os.path.join(args.root, 'res.json'), 'wt') as f:
    json.dump(df.to_dict(orient='records'), f, indent=2)
