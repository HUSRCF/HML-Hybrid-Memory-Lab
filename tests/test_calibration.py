import json
from pathlib import Path
import tempfile
import unittest

import torch

from sas_memory.calibration import load_checkpoint, microbatch_indices, restore_log, save_checkpoint


class CalibrationTests(unittest.TestCase):
    def test_cycles_preserve_order_and_example_multiplicity(self):
        self.assertEqual(microbatch_indices(64, 128, 2), [126, 127])
        self.assertEqual(microbatch_indices(65, 128, 2), [0, 1])
        flat = [i for step in range(1, 257) for i in microbatch_indices(step, 128, 2)]
        self.assertEqual(flat, list(range(128)) * 4)
        with self.assertRaises(ValueError):
            microbatch_indices(0, 128, 2)

    def test_optimizer_and_rng_resume_reproduce_next_update(self):
        torch.manual_seed(37)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        def update():
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 3)).square().sum().backward()
            optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            log = parent / "training.jsonl"
            log.write_text(json.dumps({"step": 1}) + "\n")
            checkpoint = save_checkpoint(parent, 1, model, optimizer, log, {"policy": "question"})
            update()
            expected = {name: value.clone() for name, value in model.state_dict().items()}
            load_checkpoint(checkpoint, model, optimizer)
            update()
            self.assertTrue(all(torch.equal(expected[name], value) for name, value in model.state_dict().items()))

    def test_resume_preserves_interrupted_tail_and_rejects_corrupt_prefix(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            log = parent / "training.jsonl"
            prefix = b'{"step": 1}\n'
            log.write_bytes(prefix)
            checkpoint = save_checkpoint(parent, 1, model, optimizer, log, {})
            record = json.loads((checkpoint / "checkpoint.json").read_text())
            tail = b'{"step": 2}\n'
            log.write_bytes(prefix + tail)
            retained = restore_log(log, record)
            self.assertEqual(log.read_bytes(), prefix)
            self.assertEqual(Path(retained).read_bytes(), tail)
            log.write_bytes(b'{"step": 9}\n')
            with self.assertRaisesRegex(ValueError, "prefix changed"):
                restore_log(log, record)
            with self.assertRaisesRegex(ValueError, "not be overwritten"):
                save_checkpoint(parent, 1, model, optimizer, log, {})


if __name__ == "__main__":
    unittest.main()
