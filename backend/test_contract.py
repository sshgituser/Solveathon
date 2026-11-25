"""Wire-format tests. The example data are not ML predictions."""
import json
from pathlib import Path
import unittest
from backend.contract import summarize, unavailable, validate_result

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'contracts'

class ContractTests(unittest.TestCase):
    def test_examples_match_schema_and_neutral_counts(self):
        for name in ('complete', 'incomplete', 'uncertain', 'no_pose'):
            with self.subTest(name=name):
                sample = json.loads((FIXTURES / (name + '.json')).read_text('utf-8'))
                self.assertEqual(validate_result(sample), sample)
                self.assertEqual(summarize(sample['attempts']), sample['counts'])
        self.assertEqual(unavailable(), json.loads((FIXTURES / 'unavailable.json').read_text('utf-8')))

    def test_uncertain_cannot_be_relabeled_or_counted_as_incomplete(self):
        sample = json.loads((FIXTURES / 'uncertain.json').read_text('utf-8'))
        sample['counts']['incomplete'] = 1
        with self.assertRaises(ValueError):
            validate_result(sample)
        sample['counts']['incomplete'] = 0
        sample['attempts'][0]['reason'] = None
        with self.assertRaises(ValueError):
            validate_result(sample)

    def test_nonfinite_confidence_rejected(self):
        sample = json.loads((FIXTURES / 'complete.json').read_text('utf-8'))
        sample['attempts'][0]['confidence'] = float('nan')
        with self.assertRaises(ValueError):
            validate_result(sample)
