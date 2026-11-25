"""Product-side interpretation of actual predictor outcomes (not model training)."""
import unittest

from backend.contract import validate_result
from backend.runtime import wire_result


class OutcomeTests(unittest.TestCase):
    def test_missing_pair_does_not_count_as_incomplete(self):
        data = wire_result('scratch', 'test-frozen-weights', {'label': None, 'reason': 'no_visible_forearm_pair'}, .10)
        validate_result(data)
        self.assertEqual(data['attempts'], [])
        self.assertEqual(data['counts']['total'], 0)
        self.assertEqual(data['warnings'], ['no_pose'])

    def test_partial_pair_and_low_margin_are_uncertain_for_different_reasons(self):
        for result, reason in (({'label': None, 'reason': 'incomplete_forearm_coverage'}, 'low_visibility'),
                               ({'label': 'complete', 'reason': None, 'confidence': .09}, 'low_confidence')):
            with self.subTest(reason=reason):
                data = validate_result(wire_result('scratch','test-frozen-weights',result,.10))
                self.assertEqual((data['attempts'][0]['label'],data['attempts'][0]['reason']),('uncertain',reason))
                self.assertEqual((data['counts']['complete'],data['counts']['uncertain']),(0,1))

    def test_finite_verified_decision_retains_label_and_margin(self):
        data = validate_result(wire_result('transfer','test-frozen-weights',
                               {'label':'incomplete','confidence':.30,'reason':None},.10))
        self.assertEqual(data['attempts'],[{'index':1,'label':'incomplete','confidence':.30,'reason':None}])

    def test_nonfinite_classifier_score_cannot_become_a_verdict(self):
        with self.assertRaises(ValueError):
            wire_result('scratch','test-frozen-weights',{'label':'complete','confidence':float('nan'),'reason':None},.10)
