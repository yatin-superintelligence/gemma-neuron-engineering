"""New final-type regressions only; the closed T01 v1 matrix is not repeated."""

import copy
from pathlib import Path
import tempfile
import unittest

from interpretability.durable_measurement_journal_v2 import COUNTERS, Journal, JournalError, digest


class FinalWitnessTests(unittest.TestCase):
    refusal_count = 0

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = 0

    def ready(self, assignments=0):
        self.index += 1
        observed = {'parameters': {'literal.weight': {'version': 1, 'shape': [2]}},
                    'literal_scalar': 1, 'buffers': {'literal': 'a' * 64}}
        counts = dict.fromkeys(COUNTERS, 0)
        expected = dict(counts); expected['manual_parameter_assignments'] = assignments
        spec = {'protocol_sha256': '1' * 64, 'source_sha256': '2' * 64,
                'measurement_ids': ['literal-only'], 'required_guards': ['restored'],
                'expected_counters': expected}
        j = Journal.create(Path(self.temp.name) / str(self.index), spec,
                           lambda: observed, lambda: counts)
        self.addCleanup(j.close)
        j.begin('literal-only', {'synthetic': True}, lambda: observed, lambda: counts)
        observed['parameters']['literal.weight']['version'] += assignments
        counts['manual_parameter_assignments'] = assignments
        j.prepare({'synthetic': True}, {'restored': True},
                  {'literal.weight': assignments} if assignments else {},
                  lambda: observed, lambda: counts)
        j.commit()
        return j, observed, counts

    def test_01_literal_identity_final_snapshot(self):
        j, observed, counts = self.ready()
        j.finish(lambda: observed, lambda: counts)
        self.assertEqual(j.inspect()['status'], 'complete')
        self.assertEqual(j.inspect()['terminal']['data']['observation']['state'], observed)

    def test_02_literal_declared_versions_and_actual_final_digest(self):
        j, observed, counts = self.ready(assignments=2)
        expected_hash = digest(observed)
        j.finish(lambda: observed, lambda: counts)
        self.assertEqual(j.inspect()['deltas'], {'literal.weight': 2})
        self.assertEqual(digest(j.inspect()['terminal']['data']['observation']['state']), expected_hash)
        self.assertEqual(j.inspect()['status'], 'complete')

    def test_03_new_strict_final_refusals_and_incomplete_preservation(self):
        changes = [
            ('boolean version', lambda s, c: s['parameters']['literal.weight'].update(version=True)),
            ('float version', lambda s, c: s['parameters']['literal.weight'].update(version=1.0)),
            ('float scalar', lambda s, c: s.update(literal_scalar=1.0)),
            ('float shape', lambda s, c: s['parameters']['literal.weight'].update(shape=[2.0])),
            ('boolean counter', lambda s, c: c.update(optimizer_steps=False)),
            ('float counter', lambda s, c: c.update(forward_calls=0.0)),
            ('missing counter', lambda s, c: c.pop('activation_assignments')),
            ('negative counter', lambda s, c: c.update(input_token_exposures=-1)),
            ('nonzero optimizer', lambda s, c: c.update(optimizer_steps=1)),
            ('missing final state', lambda s, c: s.clear()),
        ]
        for name, change in changes:
            with self.subTest(name=name):
                j, observed, counts = self.ready()
                change(observed, counts)
                with self.assertRaises(JournalError):
                    j.finish(lambda: observed, lambda: counts)
                type(self).refusal_count += 1
                self.assertEqual(j.inspect()['status'], 'open')
                self.assertFalse(any(e['kind'] == 'complete' for e in j.inspect()['events']))
                j.stop('Rejected synthetic final: ' + name, lambda: observed, lambda: counts)
                self.assertEqual(j.inspect()['status'], 'incomplete')
                saved = j.inspect()['terminal']['data']['observation']
                self.assertEqual(saved['state'], observed)
                self.assertEqual(saved['counters'], counts)


if __name__ == '__main__':
    unittest.main()
