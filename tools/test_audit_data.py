"""Small synthetic directory layout tests; never read or publish volunteer frames."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from tools.audit_data import audit


class AuditTests(unittest.TestCase):
    def test_conflicting_label_is_quarantined_even_if_basename_collides(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / 'Exercise' / '2_Extending the Elbow'
            for folder, payload in [('Complete', b'first'), ('Incomplete', b'second')]:
                p = base / folder / 'Train' / '04_02_00_01.mp4'
                p.parent.mkdir(parents=True)
                p.write_bytes(payload)
            result = audit(root)
            self.assertEqual(result['e02_total'], 2)
            self.assertEqual(result['splits']['train']['incomplete'], 1)
            self.assertEqual(result['quarantined']['label_conflict'], 1)
            self.assertEqual(result['unique_e02_sha256'], 2)

    def test_shared_content_across_subjects_is_rejected(self):
        with TemporaryDirectory() as temp:
            base = Path(temp) / 'Exercise' / '2_Extending the Elbow'
            for subject, source in [('01','Train'),('08','Test')]:
                p = base / 'Complete' / source / f'{subject}_02_01_01.mp4'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'same-clip')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                audit(Path(temp))
