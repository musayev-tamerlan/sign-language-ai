import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from baseline import Classifier, preprocess, fingerprint, metrics
from prepare_training import prepare
from train_baseline import train, load_split, sample_weights, class_weights
from evaluate_baseline import run
from baseline_v2 import preprocess as preprocess_v2, fingerprint as fingerprint_v2


class PipelineTests(unittest.TestCase):
    def test_hand_swap_invariance_and_missing_hand(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(1)
        a=rng.random((9,126),dtype=np.float32)
        a[2,:63]=0
        swapped=a.reshape(9,2,63)[:,::-1].reshape(9,126).copy()
        self.assertEqual(fingerprint(a),fingerprint(swapped))
        model=Classifier(3).eval()
        with torch.inference_mode():
            x=model(torch.from_numpy(preprocess(a))[None])
            y=model(torch.from_numpy(preprocess(swapped))[None])
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertTrue(np.isfinite(preprocess(a)).all())

    def test_v2_spatial_canonicalization_and_normalization(self):
        array = np.zeros((8, 126), dtype=np.float32)
        # Two identical hand shapes at different locations, supplied in opposite order.
        shape = np.zeros((21, 3), dtype=np.float32)
        shape[:, 0] = np.linspace(0, .1, 21)
        shape[:, 1] = np.linspace(0, .2, 21)
        left, right = shape.copy(), shape.copy()
        left[:, 0] += .2; right[:, 0] += .7
        array[:] = np.stack((right, left)).reshape(126)
        swapped = array.reshape(8, 2, 63)[:, ::-1].reshape(8, 126).copy()
        self.assertEqual(fingerprint_v2(array), fingerprint_v2(swapped))
        self.assertEqual(preprocess_v2(array).shape, (64, 132))
        self.assertTrue(np.isfinite(preprocess_v2(array)).all())

    def test_balanced_sampling_equalizes_class_mass(self):
        rows=[dict(label_id=0)]*20+[dict(label_id=1)]*2
        weights=sample_weights(rows,2)
        self.assertAlmostEqual(weights[:20].sum().item(),weights[20:].sum().item())
        with self.assertRaises(ValueError): sample_weights(rows,3)

    def test_soft_class_weights_are_finite_and_less_extreme(self):
        rows=[dict(label_id=0)]*100+[dict(label_id=1)]*4
        weights=class_weights(rows,2,.5)
        self.assertAlmostEqual(weights.mean().item(),1.0)
        self.assertAlmostEqual((weights[1]/weights[0]).item(),5.0,places=5)
        with self.assertRaises(ValueError): class_weights(rows,2,1.1)

    def test_metrics(self):
        m=metrics([0,0,1,1],[0,1,1,1],2)
        self.assertEqual(m['accuracy'],0.75)
        self.assertEqual(m['support'],[2,2])
        self.assertAlmostEqual(m['macro_f1'],(2/3+0.8)/2)

    def test_end_to_end_and_guardrails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); videos=root/'videos'; landmarks=root/'landmarks'
            rng=np.random.default_rng(7)
            for label in ['ANA','GÜN']:
                (videos/label).mkdir(parents=True)
                (landmarks/label).mkdir(parents=True)
                for index in range(5):
                    (videos/label/f'{index}.mp4').touch()
                    np.save(landmarks/label/f'{index}.npy',rng.random((8,126),dtype=np.float32))
            missing=videos/'ANA/missing.mp4';missing.touch()
            with self.assertRaisesRegex(ValueError,'outputs missing'):
                prepare(videos,landmarks)
            missing.unlink()
            (videos/'ANA/copy.mp4').touch()
            (landmarks/'ANA/copy.npy').write_bytes((landmarks/'ANA/0.npy').read_bytes())
            result=prepare(videos,landmarks)
            self.assertEqual(result,prepare(videos,landmarks))
            self.assertEqual(result['duplicate_groups'],1)
            hashes=[{r['sha256'] for r in rows} for rows in result['splits'].values()]
            self.assertFalse(hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
            split=root/'split.json';split.write_text(json.dumps(result),encoding='utf-8')
            checkpoint=train(split,root/'run',epochs=1,batch_size=4,hidden=8,threads=1,balanced_sampling=True)
            score=run(checkpoint,split,root/'test.json')
            self.assertEqual(sum(score['support']),2)
            self.assertTrue(checkpoint.exists())
            bad=json.loads(split.read_text());bad['splits']['test'][0]=bad['splits']['train'][0]
            split.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_split(split)


if __name__=='__main__':
    unittest.main()
