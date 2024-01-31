import os
import wget
fout = 'data.zip'
url = 'https://s3.ca-central-1.wasabisys.com/watshare/verb_envs/verbenvs_data.zip'


if os.path.isfile(fout):
    print('{} already exists!'.format(fout))
else:
    print('downloading {}'.format(fout))
    wget.download(url, out=fout)
    print('please unzip {}'.format(fout))
