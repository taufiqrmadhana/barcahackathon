"""An aligned crop cannot silently switch tracking or cached observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import video_provenance
from possession.config import Params


class VideoProvenanceContracts(unittest.TestCase):
    def fixture(self):
        class Inputs:
            game = 'TEST'
            atd = {'ball': np.array([[.2, .3, 0.]])}
            fused = np.array([[.4, .5, 0.]])

            def ball_atd(self, p):
                return dict(self.atd, ball=self.fused if p.ball_source else self.atd['ball'])
        return Inputs(), Params(ball_source='fused'), {'max_gap': 25, 'err_scale': 1., 'pan_window': 25}

    def test_legacy_cache_without_metadata_keeps_working(self):
        inputs, p, legacy = self.fixture()
        with tempfile.TemporaryDirectory() as folder, patch.object(video_provenance, 'CACHE_DIR', folder):
            self.assertIsNone(video_provenance.describe(inputs, p, 'raw', legacy))
            video_provenance.check(inputs, p, dict(kind='raw', legacy_params=legacy))

    def test_aligned_cache_rejects_changed_ball_parameters_features_or_missing_metadata(self):
        inputs, p, legacy = self.fixture()
        with tempfile.TemporaryDirectory() as folder, patch.object(video_provenance, 'CACHE_DIR', folder):
            path = video_provenance.store_path('aligned', inputs.game)
            path.write_bytes(b'features_a')
            meta = video_provenance.metadata_path('aligned', inputs.game)
            meta.write_text(json.dumps(video_provenance.metadata(inputs, p, 'aligned', legacy, 'fused')))
            provenance = video_provenance.describe(inputs, p, 'aligned', legacy)
            artifact = dict(kind='aligned', legacy_params=legacy, video_provenance=provenance)
            video_provenance.check(inputs, p, artifact)
            path.write_bytes(b'features_b')
            with self.assertRaisesRegex(ValueError, 'refit'):
                video_provenance.check(inputs, p, artifact)
            path.write_bytes(b'features_a')
            inputs.fused = inputs.fused + .1
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                video_provenance.check(inputs, p, artifact)
            inputs.fused = np.array([[.4, .5, 0.]])
            with self.assertRaisesRegex(ValueError, 'parameters'):
                video_provenance.describe(inputs, p, 'aligned', dict(legacy, max_gap=12))
            inputs.atd = {'ball': inputs.atd['ball'] + .1}
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                video_provenance.check(inputs, p, artifact)
            meta.unlink()
            with self.assertRaisesRegex(ValueError, 'refit'):
                video_provenance.check(inputs, p, artifact)


if __name__ == '__main__':
    unittest.main()
