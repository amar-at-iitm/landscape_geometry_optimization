import torch
import unittest
from torch.utils.data import DataLoader, TensorDataset

from src.landscape.visualization import get_weights, get_1d_interpolation


class VisualizationUtilsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        features = torch.randn(32, 4)
        labels = torch.randint(0, 3, (32,))
        dataset = TensorDataset(features, labels)
        self.loader = DataLoader(dataset, batch_size=16, shuffle=False)
        self.model = torch.nn.Linear(4, 3)

    def test_get_weights_returns_clones(self):
        weights = get_weights(self.model)
        weights[0].add_(10.0)
        self.assertFalse(torch.allclose(self.model.weight, weights[0]))

    def test_interpolation_restores_parameters(self):
        start_weights = get_weights(self.model)
        direction = [torch.randn_like(w) for w in start_weights]
        end_weights = [s + d for s, d in zip(start_weights, direction)]

        get_1d_interpolation(
            self.model,
            self.loader,
            start_weights,
            end_weights,
            steps=3,
            device='cpu',
        )

        for param, start in zip(self.model.parameters(), start_weights):
            self.assertTrue(torch.allclose(param.detach(), start, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
