import json
from pathlib import Path
import tempfile
import unittest
from graphcliff_pair import train
from tools.check_layer_repro_long import compare


class LongTraceComparison(unittest.TestCase):
    def test_late_gradient_difference_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = [Path(directory) / name for name in ["a", "b"]]
            rows = [dict(event="epoch_start", epoch=1), dict(event="batch", epoch=1, step=1, gradient="same"),
                    dict(event="batch", epoch=1, step=2, gradient="same"), dict(event="epoch_end", epoch=1)]
            def write(folder, records):
                folder.mkdir(exist_ok=True)
                trace = folder / "trace.jsonl"
                trace.write_text("".join(json.dumps(row)+"\n" for row in records))
                train.save_json(folder / "summary.json", dict(dataset="fixture", variant="self", epochs=1,
                                source_sha256={}, input_sha256="input", train_count=2, valid_count=1,
                                initialization="init", status="completed", optimizer_steps=2, trace_sha256=train.digest(trace)))
            for folder in folders:
                write(folder, rows)
            self.assertIsNone(compare(*folders)["first_difference"])
            changed = [dict(row) for row in rows]
            changed[2]["gradient"] = "different"
            write(folders[1], changed)
            difference = compare(*folders)["first_difference"]
            self.assertEqual((difference["step"], difference["fields"]), (2, ["gradient"]))
            for folder in folders:
                write(folder, rows[:-1])
            with self.assertRaisesRegex(AssertionError, "coverage"):
                compare(*folders)


if __name__ == "__main__":
    unittest.main()
