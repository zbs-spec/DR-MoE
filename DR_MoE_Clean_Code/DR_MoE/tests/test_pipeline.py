import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from drmoe.data import make_datasets
from drmoe.model import DRMoE, TopKAttention
from drmoe.utils import metric_scores


class MambaStub(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.layer(x)


class PipelineTests(unittest.TestCase):
    def test_data_splits(self):
        signal = np.sin(np.linspace(0, 30, 240))
        datasets, stats = make_datasets(signal, 32, 4, 0.7, 0.15, 0.1, False)
        self.assertEqual(len(datasets), 3)
        self.assertTrue(all(len(d) > 0 for d in datasets))
        self.assertIn('mean', stats)
        x, y = datasets[0][0]
        self.assertEqual(x.shape, (32,))
        self.assertEqual(y.shape, (4,))

    def test_model_forward_and_gradients(self):
        with patch('drmoe.model.Mamba', MambaStub):
            model = DRMoE(input_length=64, horizon=4, dim=32, heads=4, global_patch=16, global_stride=8, local_patch=8, local_stride=4, top_k=3)
            output, details = model(torch.randn(3, 64), return_experts=True)
            self.assertEqual(output.shape, (3, 4))
            self.assertEqual(details['weights'].shape, (3, 2))
            self.assertTrue(torch.allclose(details['weights'].sum(-1), torch.ones(3), atol=1e-6))
            output.square().mean().backward()
            self.assertIsNotNone(model.head.weight.grad)

    def test_attention(self):
        model = TopKAttention(32, 4, 3, 0.0)
        self.assertEqual(model(torch.randn(2, 8, 32)).shape, (2, 8, 32))

    def test_metrics(self):
        result = metric_scores(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(result['RMSE'], 0.0)
        self.assertEqual(result['MAE'], 0.0)


if __name__ == '__main__':
    unittest.main()
