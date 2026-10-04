"""仅合成数据/mock推理；不打开真实benchmark或最终test标签。"""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location(
    'evaluate_test_contracts', Path(__file__).resolve().parents[1] / 'tools/evaluate_test.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestLifecycleContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.csv_root = self.root / 'synthetic_data'
        self.csv_root.mkdir()
        self.output = self.root / 'evaluation'
        self.calls = []
        self.label_reads = []
        models, hashes = [], {}
        for task in ['synthetic_A', 'synthetic_B']:
            source = self.csv_root / f'{task}.csv'
            pd.DataFrame({'smiles': ['C', 'CC', 'CO'], 'split': ['train', 'test', 'test'],
                          'y': [3., 4., 5.], 'cliff_mol': [0, 0, 1]}).to_csv(source, index=False)
            hashes[task] = module.digest(source)
            pairs = self.root / 'artifacts' / 'stage' / task / 'pairs.json'
            pairs.parent.mkdir(parents=True)
            pairs.write_text(json.dumps({'train_rows': [0]}), encoding='utf-8')
            for i in range(39):
                models.append(dict(dataset=task, seed=42, arm=f'arm{i}',
                                   run_dir=f'artifacts/stage/{task}/seed42/arm{i}',
                                   checkpoint_sha256='synthetic-checkpoint'))
        self.frozen = dict(models=models, data_sha256=hashes)
        self.freeze_path = self.root / 'synthetic_freeze.json'
        self.freeze_path.write_text(json.dumps(self.frozen), encoding='utf-8')
        self.original_read = pd.read_csv
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(module, 'ROOT', self.root))
        self.stack.enter_context(patch.object(module, 'validate_freeze', return_value=self.frozen))
        self.stack.enter_context(patch.object(module, 'predict_smiles', side_effect=self.predict))
        self.stack.enter_context(patch.object(module.pd, 'read_csv', side_effect=self.read_csv))

    def read_csv(self, path, *args, **kwargs):
        if kwargs.get('usecols') == ['y', 'cliff_mol']:
            state = json.loads((self.output / 'prediction_state.json').read_text(encoding='utf-8'))
            self.assertTrue(state['labels_loaded'])
            self.assertEqual(len(state['predictions']), 78)
            self.assertEqual(self.registration()['phase'], 'labels_loaded')
            self.label_reads.append(str(path))
        return self.original_read(path, *args, **kwargs)

    def predict(self, run_dir, source, query_path, output):
        self.calls.append(str(run_dir))
        queries = self.original_read(query_path)
        result = queries.assign(query_index=[0, 1], prediction=[4.1, 4.9],
                                reference_row=[0, 0], similarity=[.5, .5], delta_prediction=[1.1, 1.9])
        result.to_csv(output, index=False)

    def registration(self):
        identity = module.freeze_identity(self.frozen)
        return json.loads((self.root / 'artifacts/test_evaluation_registry' / f'{identity}.json').read_text(encoding='utf-8'))

    def evaluate(self, **kwargs):
        module.evaluate(self.freeze_path, self.csv_root, self.output, **kwargs)

    def test_atomic_state_and_process_lock_release(self):
        path = self.root / 'state.json'
        module.save(path, {'old': 1})
        with patch.object(module.os, 'replace', side_effect=OSError('synthetic interruption')):
            with self.assertRaises(OSError):
                module.save(path, {'new': 2})
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'old': 1})
        self.assertEqual(list(self.root.glob('.state.json.*')), [])
        identity = module.freeze_identity(self.frozen)
        with module.evaluation_lock(identity):
            with self.assertRaisesRegex(RuntimeError, '并发'):
                with module.evaluation_lock(identity):
                    self.fail('second lock must fail')
        with module.evaluation_lock(identity):
            pass

    def test_complete_matrix_then_labels_and_unique_baseline(self):
        self.evaluate()
        result = json.loads((self.output / 'test_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(len(self.calls), 78)
        self.assertEqual(len(self.label_reads), 2)
        self.assertEqual(len(result['records']), 78)
        self.assertEqual(len(result['nearest_reference']), 2)
        for baseline in result['nearest_reference']:
            self.assertAlmostEqual(baseline['overall_rmse'], 2.5 ** .5)
            self.assertEqual(baseline['mae'], 1.5)
            self.assertEqual(baseline['cliff_rmse'], 2.)
            self.assertEqual(baseline['count'], 2)
        self.assertEqual(self.registration()['phase'], 'completed')
        with self.assertRaisesRegex(ValueError, '标签评估'):
            self.evaluate(resume_predictions=True)
        with self.assertRaisesRegex(ValueError, '其他输出目录'):
            module.evaluate(self.freeze_path, self.csv_root, self.root / 'second_output')
        copied = self.root / 'copied_freeze.json'
        copied.write_text(json.dumps(self.frozen, indent=4, sort_keys=True), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '其他输出目录'):
            module.evaluate(copied, self.csv_root, self.root / 'third_output')
        self.assertEqual(len(self.label_reads), 2)

    def test_unregistered_complete_prediction_is_recomputed(self):
        self._exercise_orphan(partial=False)

    def test_unregistered_partial_prediction_is_recomputed(self):
        self._exercise_orphan(partial=True)

    def _exercise_orphan(self, partial):
        real_save = module.save

        def interrupted_save(path, value):
            if Path(path).name == 'prediction_state.json' and value['predictions']:
                raise OSError('interrupted after prediction publication, before state update')
            real_save(path, value)

        with patch.object(module, 'save', side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, 'publication'):
                self.evaluate()
        self.assertEqual(self.label_reads, [])
        self.assertEqual(self.registration()['phase'], 'predictions')
        state = json.loads((self.output / 'prediction_state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['predictions'], {})
        prediction = self.output / f"{module.model_key(self.frozen['models'][0])}_predictions.csv"
        self.assertTrue(prediction.is_file())
        if partial:
            prediction.write_text('query_index,broken\n0,', encoding='utf-8')
        marker = self.output / 'user_note.txt'
        marker.write_text('preserve unrelated files', encoding='utf-8')
        self.evaluate(resume_predictions=True)
        self.assertEqual(len(self.calls), 79)
        self.assertEqual(self.calls.count(self.calls[0]), 2)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve unrelated files')
        self.assertEqual(self.registration()['phase'], 'completed')

    def test_prediction_interruption_allows_only_same_directory_resume(self):
        def interrupted(*args):
            self.predict(*args)
            raise OSError('partial prediction')

        with patch.object(module, 'predict_smiles', side_effect=interrupted):
            with self.assertRaises(OSError):
                self.evaluate()
        self.assertEqual(self.label_reads, [])
        self.assertEqual(list(self.output.glob('*_predictions.csv')), [])
        with self.assertRaisesRegex(ValueError, '其他输出目录'):
            module.evaluate(self.freeze_path, self.csv_root, self.root / 'different')
        self.evaluate(resume_predictions=True)
        self.assertEqual(len(self.label_reads), 2)

    def test_registered_prediction_tamper_is_rejected(self):
        def stop_second(*args):
            if len(self.calls) == 1:
                raise OSError('stop after one registered prediction')
            self.predict(*args)

        with patch.object(module, 'predict_smiles', side_effect=stop_second):
            with self.assertRaises(OSError):
                self.evaluate()
        prediction = next(self.output.glob('*_predictions.csv'))
        prediction.write_text('modified', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '缓存身份'):
            self.evaluate(resume_predictions=True)
        self.assertEqual(self.label_reads, [])

    def test_label_boundary_registration_precedes_state_and_fails_closed(self):
        real_save = module.save

        def interrupted_save(path, value):
            if Path(path).name == 'prediction_state.json' and value['labels_loaded']:
                raise OSError('interrupt before labels state')
            real_save(path, value)

        with patch.object(module, 'save', side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, 'labels state'):
                self.evaluate()
        self.assertEqual(self.registration()['phase'], 'labels_loaded')
        self.assertEqual(self.label_reads, [])
        with self.assertRaisesRegex(ValueError, '标签评估'):
            self.evaluate(resume_predictions=True)
        with self.assertRaisesRegex(ValueError, '其他输出目录'):
            module.evaluate(self.freeze_path, self.csv_root, self.root / 'different')

    def test_invalid_output_paths_and_reference_disagreement_fail_before_labels(self):
        with self.assertRaisesRegex(ValueError, '路径越界'):
            module.child_path(self.root, '../outside.csv')
        with self.assertRaisesRegex(ValueError, '路径字符'):
            module.model_key(dict(dataset='../outside', seed=42, arm='normal'))

        def changed_reference(*args):
            self.predict(*args)
            if len(self.calls) == 2:
                frame = self.original_read(args[-1])
                frame['reference_row'] = 1
                frame.to_csv(args[-1], index=False)

        with patch.object(module, 'predict_smiles', side_effect=changed_reference):
            with self.assertRaises(AssertionError):
                self.evaluate()
        self.assertEqual(self.label_reads, [])
        self.assertEqual(self.registration()['phase'], 'predictions')


if __name__ == '__main__':
    unittest.main()
