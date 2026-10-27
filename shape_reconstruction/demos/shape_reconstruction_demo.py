"""Live or recorded reconstruction: traditional, neural, or same-frame comparison."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ..config import DEFAULT_CONFIG_PATH, load_config
from ..visualization.comparison import export_surfaces, write_ply


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['traditional','neural','compare'],default='traditional')
    p.add_argument('--model_path')
    p.add_argument('--device')
    p.add_argument('--config',default=str(DEFAULT_CONFIG_PATH))
    p.add_argument('--auto_ref',action='store_true')
    p.add_argument('--replay',help='Recorded .npz file/directory or collected data root; no camera required')
    p.add_argument('--loop',action='store_true',help='Repeat recorded frames for a defense demonstration')
    p.add_argument('--headless',action='store_true',help='Export recorded results without a GUI')
    p.add_argument('--output',default='demo_results')
    p.add_argument('--display_max_mm',type=float,default=3.0,help='Shared fixed display scale, mm')
    return p


def save_result(out, stem, diff, maps, names, mm_per_pixel, display_max_mm, raw=None, ref=None):
    out.mkdir(parents=True,exist_ok=True)
    payload={'diff':diff, 'mm_per_pixel':mm_per_pixel}
    if raw is not None: payload['raw_gray']=raw
    if ref is not None: payload['reference_gray']=ref
    payload.update(dict(zip(names,maps)))
    np.savez_compressed(out/f'{stem}.npz',**payload)
    export_surfaces(out/f'{stem}_3d.png',maps,names,mm_per_pixel,display_max_mm)
    for height,name in zip(maps,names):
        write_ply(out/f'{stem}_{name}.ply',height,mm_per_pixel)
    print(f'Saved {out/stem} (.npz, 3D PNG, PLY), units mm')


def main():
    p=build_parser();args=p.parse_args()
    if args.headless and not args.replay: p.error('--headless requires --replay')
    if args.headless and args.loop: p.error('--loop is only for interactive replay')
    if args.display_max_mm<=0: p.error('--display_max_mm must be positive')
    model=None
    if args.mode in ('neural','compare'):
        if not args.model_path: p.error('--model_path is required for neural/compare')
        from ..learning.inference import ReconstructionModel
        model=ReconstructionModel(args.model_path,args.device)
    names = ['traditional','neural'] if args.mode=='compare' else [args.mode]
    sensor, viewer = None, None
    files=[];data_root=None;metadata={}
    if args.replay:
        source=Path(args.replay)
        if (source/'diff_images').is_dir():
            data_root=source
            files=sorted((source/'diff_images').glob('*.npy'))
            if (source/'samples.jsonl').is_file():
                metadata={r['file']:r for r in (json.loads(line) for line in (source/'samples.jsonl').read_text(encoding='utf-8').splitlines())}
            cfg=load_config(args.config)
            camera=cfg['camera_calibration']
            calibration=Path(cfg['calibration_root_dir'])/f"sensor_{cfg['sensor_id']}"
            fallback_scale=float(np.load(str(calibration)+camera['camera_calibration_dir']+camera['position_scale_path'],allow_pickle=False)[2])
        else:
            files=[source] if source.is_file() else sorted(source.glob('*.npz'))
        if not files: p.error('No recorded frames found')
    else:
        from ..core.sensor import Sensor
        sensor=Sensor(load_config(args.config),auto_ref=args.auto_ref)
    out=Path(args.output)
    frame=0
    if not args.headless: import cv2
    try:
        while True:
            raw=ref=None
            traditional=None
            if files:
                if frame>=len(files) and not args.loop: break
                path=files[frame % len(files)]
                if data_root is not None:
                    diff=np.load(path,allow_pickle=False).astype(np.float32)
                    raw=np.load(data_root/'raw_images'/path.name,allow_pickle=False)
                    ref=np.load(data_root/'reference_images'/path.name,allow_pickle=False)
                    mm_per_pixel=float(metadata.get(path.name,{}).get('mm_per_pixel',fallback_scale))
                    if args.mode!='neural':
                        from ..learning.evaluation import traditional_from_saved
                        traditional=traditional_from_saved(data_root,path.name,args.config)
                else:
                    with np.load(path,allow_pickle=False) as record:
                        diff=record['diff'].astype(np.float32)
                        mm_per_pixel=float(record['mm_per_pixel'])
                        traditional=record['traditional'].copy() if 'traditional' in record else None
                        if 'raw_gray' in record: raw=record['raw_gray'].copy()
                        if 'reference_gray' in record: ref=record['reference_gray'].copy()
                traditional_ms=None
                if args.mode!='neural' and traditional is None:
                    raise ValueError('Replay lacks traditional height; record a traditional or compare frame')
            else:
                img=sensor.get_rectify_crop_image()
                raw=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ref=sensor.ref_GRAY
                diff=ref.astype(np.float32)-raw.astype(np.float32)
                mm_per_pixel=float(sensor.pixel_per_mm)  # Historical attribute is actually mm/pixel.
                t0=time.perf_counter()
                if args.mode!='neural': traditional=sensor.raw_image_2_height_map(raw)
                traditional_ms=(time.perf_counter()-t0)*1000
            if not np.isfinite(mm_per_pixel) or mm_per_pixel<=0: raise ValueError('Invalid physical pixel scale')
            if diff.ndim!=2 or not np.isfinite(diff).all(): raise ValueError('Expected a finite 2D difference image')
            maps=[];timings=[]
            for name in names:
                if name=='traditional':
                    height=traditional;elapsed=traditional_ms
                else:
                    if frame==0: model.predict(diff)  # Warm-up is excluded from timing.
                    t0=time.perf_counter();height=model.predict(diff);elapsed=(time.perf_counter()-t0)*1000
                if height.shape!=diff.shape or not np.isfinite(height).all(): raise ValueError('Invalid reconstruction')
                maps.append(height);timings.append(elapsed)
            if args.headless:
                save_result(out,f'frame_{frame:04d}',diff,maps,names,mm_per_pixel,args.display_max_mm,raw,ref)
            else:
                from ..visualization.comparison import ComparisonVisualizer
                if viewer is None: viewer=ComparisonVisualizer(maps,mm_per_pixel,args.display_max_mm,names)
                if not viewer.update(maps): break
                panels=[]
                original=cv2.cvtColor(np.clip(raw if raw is not None else diff,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                cv2.putText(original,'Input | s:save q:quit',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                panels.append(original)
                for height,name,ms in zip(maps,names,timings):
                    panel=cv2.applyColorMap(np.clip(height/args.display_max_mm*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
                    latency='recorded' if ms is None else f'{ms:.1f}ms'
                    cv2.putText(panel,f'{name}: {height.max():.3f}mm {latency}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    panels.append(panel)
                display=np.hstack(panels)
                cv2.imshow('Input | Traditional | Neural; shared color scale',display)
                key=cv2.waitKey(250 if files else 1)&255
                if key==ord('q'): break
                if key==ord('s'):
                    stem=f'frame_{time.time_ns()}'
                    save_result(out,stem,diff,maps,names,mm_per_pixel,args.display_max_mm,raw,ref)
                    cv2.imwrite(str(out/f'{stem}_2d.png'),display)
            frame+=1
    finally:
        if sensor is not None: sensor.cap.release()
        if viewer is not None: viewer.close()
        if not args.headless: cv2.destroyAllWindows()


if __name__=='__main__':
    main()
