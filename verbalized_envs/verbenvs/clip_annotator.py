import os
import torch
import numpy as np


class CLIPAnnotator:

    @classmethod
    def load_nps(cls, fnps):
        tmp = torch.load(fnps, map_location=torch.device('cpu'))
        seen = set()
        nps, nps_emb = [], []
        for t, e in zip(tmp['texts'], tmp['emb']):
            t = t.lower().strip()
            if len(t) > 1 and t not in seen:
                nps.append(t)
                nps_emb.append(e)
                seen.add(t)
        return nps, nps_emb

    @classmethod
    def scan(cls, dpano_emb, texts, pano, nps, nps_emb, device=torch.device('cpu')):
        texts = list(set([t.lower() for t in texts]))
        indices = [nps.index(t.lower()) for t in texts if t in nps]
        embs = [nps_emb[i] for i in indices]
        with torch.no_grad():
            text_emb = torch.stack(embs, dim=0).to(device)
            tmp = torch.load(os.path.join(dpano_emb, '{}.pt'.format(pano)), map_location=torch.device('cpu'))
            pano_emb = tmp['emb'].to(device)
            pano_info = tmp['patch_info']

            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            pano_emb = pano_emb / pano_emb.norm(p=2, dim=-1, keepdim=True)
            scores = torch.matmul(text_emb, pano_emb.t())

        out = []
        for i, p in enumerate(pano_info):
            scores_i = scores[:, i].tolist()
            out.append(dict(
                i=p['i'],
                j=p['j'],
                step_size=300,
                scores=sorted(list(zip(texts, scores_i)), key=lambda tup: tup[1], reverse=True)
            ))
        return out

    @classmethod
    def embed_image(cls, image, model, processor, step_size=300, batch_size=1024):
        from patchify import patchify
        patch_shape = (step_size, step_size, 3)
        img_arr = np.asarray(image)
        patches = patchify(img_arr, patch_shape, step_size)

        images = []
        patch_info = []
        for i, row in enumerate(patches):
            for j, patch in enumerate(row):
                i_start = i * step_size
                j_start = j * step_size
                y = img_arr[i_start:i_start+step_size, j_start:j_start+step_size]
                check = patch[0] == y
                assert check.sum().item() == check.size
                images.append(torch.from_numpy(patch[0]))
                patch_info.append(dict(i=i_start, j=j_start))
        with torch.no_grad():
            emb = []
            for i in range(0, len(images), batch_size):
                inputs = processor(images=images[i:i+batch_size], return_tensors="pt", padding=True).to(model.device)
                emb.append(model.get_image_features(**inputs))
            emb = torch.cat(emb, dim=0)
        return dict(emb=emb, patch_info=patch_info)

    @classmethod
    def embed_text(cls, texts, model, tokenizer, batch_size=10000):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b = [t.strip() for t in texts[i:i+batch_size]]
                inputs = tokenizer(b, padding=True, return_tensors="pt").to(model.device)
                out.append(model.get_text_features(**inputs))
        emb = torch.cat(out, dim=0)
        return dict(emb=emb, texts=texts)
