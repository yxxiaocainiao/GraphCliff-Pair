import unittest
import pandas as pd

from tools.report_diagnostics import nearest_reference, check_metrics, reconstruct_weights, check_weight_history


class DiagnosticContracts(unittest.TestCase):
    def test_reference_predictions_do_not_depend_on_query_labels(self):
        frame = pd.DataFrame({'y': [1., 3., 2.], 'cliff_mol': [0, 1, 0]})
        pairs = [dict(query=0, reference=2), dict(query=1, reference=2)]
        result = nearest_reference(frame, pairs)
        self.assertEqual(result['overall_rmse'], 1.)
        self.assertEqual(result['prediction_std'], 0.)
        changed = dict(result, cliff_rmse=0.)
        with self.assertRaisesRegex(AssertionError, 'Top-1'):
            check_metrics(result, changed, 'Top-1')

    def test_weight_reconstruction_and_history_tamper(self):
        frame = pd.DataFrame({'y': [0., 1., 10.]})
        pairs = [dict(query=1, reference=0), dict(query=2, reference=0)]
        config = dict(alpha_max=1., warmup_epochs=2, weight_cap=3., batch_size=2)
        rows = reconstruct_weights(frame, pairs, config, 'dynamic', 42, 1., 2)
        # Epoch1 unnormalized weights 1.5,2.5; epoch2 2,4.
        self.assertAlmostEqual(rows[0]['min'], .75)
        self.assertAlmostEqual(rows[0]['max'], 1.25)
        self.assertAlmostEqual(rows[1]['min'], 2/3, places=6)
        self.assertAlmostEqual(rows[1]['max'], 4/3, places=6)
        self.assertAlmostEqual(rows[1]['mean'], 1.)
        history = [dict(epoch=r['epoch'], weight_min=r['min'], weight_max=r['max']) for r in rows]
        check_weight_history(rows, history)
        history[1]['weight_max'] = 5.
        with self.assertRaisesRegex(AssertionError, 'history'):
            check_weight_history(rows, history)
        mse = reconstruct_weights(frame, pairs, config, 'mse', 42, 1., 2)
        self.assertTrue(all(r['min'] == r['max'] == 1. for r in mse))
