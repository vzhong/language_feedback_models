import os
import tqdm
import glob
import json
import spacy
import torch
import argparse
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
from verbenvs.clip_annotator import CLIPAnnotator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/pretrained/clip-vit-large-patch14', help='CLIP model to use')
    parser.add_argument('--dpanos', default='/data/touchdown/jpegs_manhattan_touchdown_2022', help='original touchdown panoramas')
    parser.add_argument('--dtrajs', default='/data/touchdown/raw_traj', help='original touchdown trajectories')
    parser.add_argument('--dout', default='/data/touchdown/preprocess', help='output folder')
    parser.add_argument('--spacy_model', default='en_core_web_lg', help='spacy model to use for parsing NPs')
    parser.add_argument('--text_batch', default=128, help='batch size for encoding text')
    parser.add_argument('--image_batch', default=128, help='batch size for encoding images')
    parser.add_argument('--step_size', default=300, help='patch dimension for CLIP representations')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    dout_panos = os.path.join(args.dout, 'panos')
    if not os.path.isdir(dout_panos):
        os.makedirs(dout_panos)

    print('loading CLIP models')
    model = AutoModel.from_pretrained(args.model).cuda()
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    fnps = os.path.join(args.dout, 'nps.pt')
    if not os.path.isfile(fnps):
        all_nps = set()
        print('parsing for NPs')
        sp = spacy.load(args.spacy_model)
        for fname in tqdm.tqdm(glob.glob(os.path.join(args.dtrajs, '*.json')), desc='parsing trajectories'):
            fout = os.path.join(args.dout, os.path.basename(fname))
            with open(fname) as f:
                data = []
                for line in tqdm.tqdm(f, desc='example'):
                    ex = json.loads(line)
                    nps = set()
                    for n in sp(ex['navigation_text']).noun_chunks:
                        nps.add(n.text.lower())
                    ex['noun_chunks'] = list(nps)
                    all_nps |= nps
                    data.append(ex)
            print('writing to {}'.format(fout))
            with open(fout, 'wt') as f:
                json.dump(data, f, indent=2)
        all_nps = sorted(list(all_nps))

        print('embedding {} NPs'.format(len(all_nps)))
        enc = CLIPAnnotator.embed_text(all_nps, model, tokenizer, batch_size=args.text_batch)
        torch.save(enc, fnps)

    for fname in tqdm.tqdm(glob.glob(os.path.join(args.dpanos, '*.jpg')), desc='featurizing panos'):
        fout = os.path.join(dout_panos, os.path.basename(fname).replace('.jpg', '.pt'))
        if not os.path.isfile(fout):
            enc = CLIPAnnotator.embed_image(Image.open(fname), model, processor, step_size=args.step_size, batch_size=args.image_batch)
            torch.save(enc, fout)
