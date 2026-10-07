import unittest
from unittest.mock import patch
import torch
import graphcliff_pair.model as shared
from graphcliff_pair import train
from experiments.mechanism_retry import repro


class RetryReproTest(unittest.TestCase):
    def test_tied_maxima_and_runner_restoration(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            x = torch.tensor([[2., -3.], [2., -4.], [1., -1.], [-2., 5.]], device=device, requires_grad=True)
            batch = torch.tensor([0, 0, 1, 1], device=device)
            value = repro.stable_max_pool(x, batch)
            self.assertTrue(torch.equal(value, torch.tensor([[2., -3.], [1., 5.]], device=device)))
            value.sum().backward()
            self.assertTrue(torch.equal(x.grad, torch.tensor([[.5, 1.], [.5, 0.], [1., 0.], [0., 1.]], device=device)))
        original_pool, original_save = shared.global_max_pool, train.save_json
        def interrupted(*args):
            self.assertIs(shared.global_max_pool, repro.stable_max_pool)
            raise RuntimeError("intentional interruption")
        with patch.object(repro, "original_run", interrupted):
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                repro.run(None, None, None)
        self.assertIs(shared.global_max_pool, original_pool)
        self.assertIs(train.save_json, original_save)


if __name__ == "__main__":
    unittest.main()
