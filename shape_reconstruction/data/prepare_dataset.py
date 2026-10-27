"""Screen every pair and create reproducible splits without moving source data."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from .pairs import load_pair


def prepare(data_dir, output, seed=42, val_fraction=.15, test_fraction=.15,
            height_scale=10.0, group_by='contact', legacy_block_size=None):
    root, output = Path(data_dir).resolve(), Path(output).resolve()
    if not np.isfinite([val_fraction,test_fraction,height_scale]).all() or height_scale <= 0:
        raise ValueError('Fractions and height scale must be finite and scale positive')
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError('Validation/test fractions must be positive and sum to < 1')
    if group_by not in ('contact','object'):
        raise ValueError('group_by must be contact or object')
    metadata = {}
    if (root/'samples.jsonl').is_file():
        for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['file'] in metadata:
                raise ValueError(f'Duplicate metadata for {row["file"]}')
            metadata[row['file']] = row
    missing, invalid, rejected, accepted, seen = [], [], [], [], {}
    shape = None
    files = sorted((root/'diff_images').glob('*.npy'))
    for index, path in enumerate(files):
        name = path.name
        if not (root/'height_maps'/name).is_file():
            missing.append(name); continue
        try:
            diff,height = load_pair(root,name,height_scale)
            if shape is not None and diff.shape != shape:
                raise ValueError(f'Mixed dimensions: expected {shape}, got {diff.shape}')
            shape = diff.shape
        except (ValueError,OSError,EOFError) as exc:
            invalid.append({'file':name,'reason':str(exc)}); continue
        item = metadata.get(name,{})
        contact = bool(height.max() > .05)
        if (np.abs(diff) >= 254).mean() > .05:
            rejected.append({'file':name,'reason':'difference_saturation_over_5_percent'}); continue
        if contact and diff.std() < .01:
            rejected.append({'file':name,'reason':'contact_label_with_constant_input'}); continue
        # Exact input duplicates cannot straddle splits, even under different filenames.
        digest = hashlib.sha256(np.ascontiguousarray(diff).tobytes()).hexdigest()
        if digest in seen:
            rejected.append({'file':name,'reason':'duplicate_input','duplicate_of':seen[digest]}); continue
        seen[digest] = name
        if group_by == 'object':
            group = item.get('object_id')
            if not group or group == 'unknown':
                raise ValueError(f'{name}: object split requires --object_id metadata for every sample')
        else:
            group = item.get('group')
            if not group:
                if not legacy_block_size or legacy_block_size < 2:
                    raise ValueError('Missing contact metadata; recollect or explicitly use --legacy_block_size N. Legacy blocks cannot establish independent contacts.')
                group = f'legacy_block_{index//legacy_block_size}'
        accepted.append({'file':name,'group':group,'object_id':item.get('object_id','unknown'),
                         'label_source':item.get('label_source','unknown'),'input_sha256':digest,
                         'max_height_mm':float(height.max()),'mean_height_mm':float(height.mean()),
                         'contact_pixels':int((height>.05).sum()),'is_background':not contact,
                         'diff_std':float(diff.std())})
    if not accepted:
        raise ValueError('No valid paired samples found')
    groups = defaultdict(list)
    for row in accepted: groups[row['group']].append(row)
    keys = sorted(groups); np.random.default_rng(seed).shuffle(keys)
    n = len(keys)
    if n < 3: raise ValueError('Need at least three independent groups for train/val/test; collect more separate contacts/objects')
    n_test = min(max(1,round(n*test_fraction)),n-2)
    n_val = min(max(1,round(n*val_fraction)),n-n_test-1)
    splits = {'train':[],'val':[],'test':[]}
    for i,key in enumerate(keys):
        split='test' if i<n_test else ('val' if i<n_test+n_val else 'train')
        splits[split].extend(row['file'] for row in groups[key])
    for values in splits.values(): values.sort()
    output.parent.mkdir(parents=True,exist_ok=True)
    orphan_heights=sorted(p.name for p in (root/'height_maps').glob('*.npy') if not (root/'diff_images'/p.name).exists())
    document={'version':2,'data_dir':os.path.relpath(root,output.parent),'height_scale_mm':height_scale,
              'seed':seed,'group_by':group_by,'legacy_block_size':legacy_block_size,
              'label_source':sorted({r['label_source'] for r in accepted}),'splits':splits,'samples':accepted,
              'screening':{'accepted':len(accepted),'missing_height':missing,'orphan_height':orphan_heights,
                           'invalid':invalid,'rejected':rejected}}
    # A fixed manifest is an experiment boundary; regenerate under a new name deliberately.
    with output.open('x',encoding='utf-8') as stream:
        json.dump(document,stream,indent=2,ensure_ascii=False)
    print(f'Kept {len(accepted)}/{len(files)} samples; {n} groups; train/val/test = '+ '/'.join(str(len(splits[k])) for k in splits))
    print(f'Label source: {document["label_source"]}; manifest: {output}')
    if legacy_block_size: print('WARNING: temporal blocks are weaker than annotated contact/object separation.')
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',required=True)
    p.add_argument('--output',default='reconstruction_data/splits.json')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val_fraction',type=float,default=.15)
    p.add_argument('--test_fraction',type=float,default=.15)
    p.add_argument('--height_scale',type=float,default=10.)
    p.add_argument('--group_by',choices=['contact','object'],default='contact')
    p.add_argument('--legacy_block_size',type=int)
    a=p.parse_args()
    prepare(a.data_dir,a.output,a.seed,a.val_fraction,a.test_fraction,a.height_scale,a.group_by,a.legacy_block_size)


if __name__=='__main__': main()
