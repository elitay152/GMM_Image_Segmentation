import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from skimage import io

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'TP02_Unsupervised_GMM_Demo.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns = {}
        for i in (1, 3, 5):
            exec(''.join(CELLS[i]['source']), self.ns)

    def test_image_formats(self):
        with tempfile.TemporaryDirectory() as d:
            gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            rgba = np.concatenate([rgb, np.full((16, 16, 1), 255, dtype=np.uint8)], axis=2)
            for name, arr in [('gray', gray), ('rgb', rgb), ('rgba', rgba)]:
                path = Path(d) / (name + '.png')
                io.imsave(path, arr)
                result = self.ns['load_image_as_array'](path)
                np.testing.assert_allclose(result, gray / 255)

    def test_clusters_and_invalid_inputs(self):
        rng = np.random.default_rng(42)
        image = np.tile(np.repeat(np.arange(5), 20), (30, 1)) + rng.normal(0, .02, (30, 100))
        segment = self.ns['segment_image_with_gmm']
        labels = segment(image, 5)
        self.assertEqual(labels.shape, image.shape)
        np.testing.assert_array_equal(np.unique(labels), np.arange(5))
        self.assertTrue(np.all(np.diff([image[labels == i].mean() for i in range(5)]) > 0))
        for image, count in [(np.zeros((2, 2)), 5), (np.empty((0, 2)), 1),
                             (np.array([[np.nan]]), 1), (np.ones((2, 2, 3)), 1),
                             (np.ones((2, 2)), 0)]:
            with self.assertRaises(ValueError):
                segment(image, count)

    def test_missing_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as d:
            for path in (d, str(Path(d) / 'missing')):
                with patch.dict(os.environ, {'GMM_IMAGE_DIR': path}):
                    with self.assertRaises(FileNotFoundError):
                        exec(''.join(CELLS[10]['source']), self.ns)

    def test_complete_notebook(self):
        import nbformat
        from nbclient import NotebookClient
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / 'nested'
            nested.mkdir()
            image = np.tile(np.arange(160, dtype=np.uint8), (64, 1))
            io.imsave(nested / 'gray.PNG', image)
            nb = nbformat.read(NOTEBOOK, as_version=4)
            nb.cells[1].source = '%matplotlib inline\n' + nb.cells[1].source
            with patch.dict(os.environ, {'GMM_IMAGE_DIR': d}):
                NotebookClient(nb, timeout=120, kernel_name='python3',
                               resources={'metadata': {'path': d}}).execute()
            self.assertEqual(len(nb.cells[10].outputs) > 0, True)
            self.assertTrue(any('data' in o for o in nb.cells[14].outputs))


if __name__ == '__main__':
    unittest.main()
