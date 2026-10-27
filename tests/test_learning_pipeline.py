"""Offline checks for geometry, model dimensions, normalization and leakage-free splits."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from shape_reconstruction.learning.model import AttentionResidualUNet
from shape_reconstruction.learning.inference import ReconstructionModel
from shape_reconstruction.learning.ball_reference import spherical_cap_reference
from shape_reconstruction.data.prepare_dataset import prepare


class PipelineTests(unittest.TestCase):
    def test_attention_unet_keeps_300x400_sensor_resolution(self):
        model = AttentionResidualUNet(base_features=8)
        result = model(torch.rand(1, 1, 300, 400))
        self.assertEqual(result.shape, (1, 1, 300, 400))
        result.mean().backward()

    def test_spherical_cap_uses_mm_per_pixel(self):
        height = spherical_cap_reference((101, 101), (50, 50), 10, 5, .1)
        self.assertEqual(height.shape, (101, 101))
        self.assertAlmostEqual(float(height.max()), 5 - np.sqrt(25 - 1), places=5)
        self.assertEqual(float(height[50, 50]), float(height.max()))
        self.assertEqual(float(height[50, 61]), 0.)

    def test_group_split_and_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'data'
            for folder in ('diff_images','height_maps'):
                (root/folder).mkdir(parents=True)
            with (root/'samples.jsonl').open('w') as f:
                for i in range(10):
                    x=np.zeros((32,48),np.float32); x[8:24,10+i%5:20+i%5]=100
                    y=np.zeros_like(x); y[8:24,10+i%5:20+i%5]=1+i*.01
                    np.save(root/'diff_images'/f'{i:03}.npy',x)
                    np.save(root/'height_maps'/f'{i:03}.npy',y)
                    f.write(json.dumps({'file':f'{i:03}.npy','group':f'touch{i//2}',
                                        'label_source':'traditional_pseudolabel'})+'\n')
            manifest=Path(td)/'splits.json'
            doc=prepare(root,manifest,seed=2)
            by_name={x['file']:x['group'] for x in doc['samples']}
            split_groups=[{by_name[n] for n in doc['splits'][k]} for k in ('train','val','test')]
            self.assertFalse(split_groups[0]&split_groups[1])
            self.assertFalse(split_groups[0]&split_groups[2])
            model=AttentionResidualUNet(base_features=8)
            checkpoint=Path(td)/'model.pth'
            torch.save({'model_type':'attention_unet','base_features':8,
                        'model_state_dict':model.state_dict(),'height_scale':10.,
                        'input_mode':'positive_diff'},checkpoint)
            inference=ReconstructionModel(checkpoint,'cpu')
            self.assertEqual(inference.predict(np.zeros((35,49),np.float32)).shape,(35,49))


if __name__=='__main__':
    unittest.main()
