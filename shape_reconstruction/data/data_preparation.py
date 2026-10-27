"""Collect stable touches with traditional pseudo-labels and replayable raw frames."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
import cv2
import numpy as np
from ..config import DEFAULT_CONFIG_PATH, load_config
from ..core.sensor import Sensor
from .pairs import validate_pair


def get_existing_count(save_dir):
    return len(list((Path(save_dir)/'diff_images').glob('*.npy')))


def next_sample_index(save_dir):
    indices=[int(p.stem) for folder in ('diff_images','height_maps','raw_images','reference_images')
             for p in (Path(save_dir)/folder).glob('*.npy') if p.stem.isdigit()]
    return max(indices,default=-1)+1


def save_sample(root,index,diff,height,raw,reference,metadata):
    validate_pair(diff,height)
    stem=f'{index:06d}' if isinstance(index,int) else str(index)
    if Path(stem).name!=stem or stem in ('','.','..') or any(c in stem for c in '/\\:'):
        raise ValueError('Sample must be a safe filename stem')
    if raw.shape!=diff.shape or reference.shape!=diff.shape or raw.dtype!=np.uint8 or reference.dtype!=np.uint8:
        raise ValueError('Raw/reference must be matching uint8 grayscale arrays')
    name=f'{stem}.npy';written=[]
    try:
        for folder,array in (('diff_images',diff),('height_maps',height),
                             ('raw_images',raw),('reference_images',reference)):
            path=Path(root)/folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                written.append(path)
                np.save(stream,array,allow_pickle=False)
        row={'file':name,**metadata}
        with (Path(root)/'samples.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    except Exception:
        for path in written:path.unlink(missing_ok=True)
        raise
    return name


def collect_data_from_camera(sensor,save_dir='reconstruction_data/raw',max_samples=None,
                             min_interval=.5,min_input_change=2.,min_height_change=.03,
                             object_id='unknown',stability_threshold=1.5,background_interval=10.):
    if max_samples is not None and max_samples<=0:raise ValueError('max_samples must be positive')
    if min_interval<0 or stability_threshold<=0 or background_interval<=0:raise ValueError('Invalid thresholds')
    root=Path(save_dir);root.mkdir(parents=True,exist_ok=True)
    index=next_sample_index(root);new=0;session=str(uuid.uuid4())
    touch_id=None;touching=False;released_frames=0;stable_frames=0
    last_diff=last_height=previous=None
    last_save=0.;last_background=0.
    print('s: save stable frame; q: finish. Release the object completely between touches.')
    try:
        while sensor.cap.isOpened() and (max_samples is None or new<max_samples):
            image=sensor.get_rectify_crop_image();gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            diff=sensor.ref_GRAY.astype(np.float32)-gray.astype(np.float32)
            height=Sensor.raw_image_2_height_map(sensor,gray).astype(np.float32)
            validate_pair(diff,height)
            peak=float(height.max())
            if peak>.05 and not touching:
                touching=True;touch_id=f'{session}:{uuid.uuid4()}'
                last_diff=last_height=None
            if peak<.025:
                released_frames+=1
                if released_frames>=3:touching=False
            else:released_frames=0
            stable_frames=stable_frames+1 if previous is not None and np.mean(np.abs(diff-previous))<=stability_threshold else 0
            previous=diff.copy()
            stable=stable_frames>=3
            preview=np.hstack([image,cv2.cvtColor(sensor.height_map_2_depth_map(height),cv2.COLOR_GRAY2BGR)])
            cv2.putText(preview,f'New {new} | {object_id} | peak {peak:.3f}mm | stable {stable} | s:save q:quit',
                        (10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow('Acquisition: input | traditional pseudo-label',preview)
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            now=time.monotonic()
            changed=last_diff is None or np.mean(np.abs(diff-last_diff))>=min_input_change
            if last_height is not None:
                mask=(height>.05)|(last_height>.05)
                changed=changed or (mask.any() and np.mean(np.abs(height-last_height)[mask])>=min_height_change)
            background=not touching and peak<=.025 and now-last_background>=background_interval
            save=stable and now-last_save>=min_interval and (key==ord('s') or (touching and changed) or background)
            if not save:continue
            label_group=touch_id if touching else f'{session}:background'
            metadata={'group':label_group,'session':session,'object_id':object_id,
                      'time':datetime.now(timezone.utc).isoformat(),'label_source':'traditional_pseudolabel',
                      'max_height_mm':peak,'is_background':not touching,'mm_per_pixel':float(sensor.pixel_per_mm),
                      'lighting_threshold':sensor.lighting_threshold,'kernel_list':list(sensor.kernel_list)}
            save_sample(root,index,diff,height,gray,sensor.ref_GRAY,metadata)
            index+=1;new+=1;last_save=now
            if not touching:last_background=now
            last_diff,last_height=diff.copy(),height.copy()
            print(f'Saved {index-1:06d}: {peak:.3f}mm; new={new}')
    finally:cv2.destroyAllWindows()
    return new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG_PATH))
    p.add_argument('--output_dir',default='reconstruction_data/raw')
    p.add_argument('--object_id',required=True,help='Actual object identity, e.g. ball_R4mm or key_A')
    p.add_argument('--max_samples',type=int)
    p.add_argument('--auto_ref',action='store_true')
    p.add_argument('--stability_threshold',type=float,default=1.5)
    p.add_argument('--min_interval',type=float,default=.5)
    a=p.parse_args();sensor=Sensor(load_config(a.config),auto_ref=a.auto_ref)
    try:
        total=collect_data_from_camera(sensor,a.output_dir,a.max_samples,min_interval=a.min_interval,
                                       object_id=a.object_id,stability_threshold=a.stability_threshold)
        print(f'Collected {total} new pairs in {a.output_dir}')
    finally:sensor.cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
