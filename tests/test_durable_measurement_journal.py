"""New synthetic host checks for T01; no pretrained model or research records."""

import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from interpretability.durable_measurement_journal import COUNTERS, Journal, JournalError, canonical, digest


def state():
    return {'parameters': {'literal.weight': {'version': 1, 'shape': [2], 'dtype': 'literal'}},
            'buffers': {'literal': 'a' * 64}, 'rng': 'b' * 64, 'hooks': []}


def counters():
    return dict.fromkeys(COUNTERS, 0)


def contract():
    expected = counters()
    expected.update(forward_calls=6, input_token_exposures=10,
                    manual_parameter_assignments=4, activation_assignments=6)
    return {'protocol_sha256': '1' * 64, 'source_sha256': '2' * 64,
            'measurement_ids': ['literal-A', 'literal-B'],
            'required_guards': ['native', 'restored'], 'expected_counters': expected}


CHILD = r'''
import json, os, sys
from pathlib import Path
from interpretability.durable_measurement_journal import Journal
root, mode = Path(sys.argv[1]), sys.argv[2]
spec, value, counts = map(json.loads, sys.argv[3:])
j = Journal.create(root, spec, lambda: value, lambda: counts)
j.begin('literal-A', {'context_id':'literal-only'}, lambda: value, lambda: counts)
if mode == 'prepared':
 value['parameters']['literal.weight']['version'] += 2
 counts.update(forward_calls=3,input_token_exposures=5,manual_parameter_assignments=2,activation_assignments=3)
 j.prepare({'literal_metric':7}, {'native':True,'restored':True}, {'literal.weight':2}, lambda:value, lambda:counts)
elif mode == 'uncommitted':
 j.con.execute('BEGIN IMMEDIATE')
 j.con.execute('INSERT INTO events VALUES(2, ?, ?)', ('{"unfinished":true}', 'not-a-digest'))
os._exit(0)
'''


class DurableJournalTests(unittest.TestCase):
    refusal_count = 0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'literal-journal'
        self.value, self.counts = state(), counters()
        self.captures = [0, 0]

    def read_state(self):
        self.captures[0] += 1
        return self.value

    def read_counters(self):
        self.captures[1] += 1
        return self.counts

    def create(self):
        j = Journal.create(self.root, contract(), self.read_state, self.read_counters)
        self.addCleanup(j.close)
        return j

    def refuse(self, call, exception=JournalError):
        with self.assertRaises(exception):
            call()
        type(self).refusal_count += 1

    def work(self):
        self.value['parameters']['literal.weight']['version'] += 2
        for name, increment in [('forward_calls', 3), ('input_token_exposures', 5),
                                ('manual_parameter_assignments', 2), ('activation_assignments', 3)]:
            self.counts[name] += increment

    def prepare(self, j):
        return j.prepare({'literal_metric': 7}, {'native': True, 'restored': True},
                         {'literal.weight': 2}, self.read_state, self.read_counters)

    def child(self, mode):
        subprocess.run([sys.executable, '-c', CHILD, str(self.root), mode,
                        canonical(contract()), canonical(state()), canonical(counters())],
                       check=True, timeout=20, capture_output=True, text=True)

    def test_01_complete_observed_state_counters_and_capture_order(self):
        j = self.create()
        for name in ['literal-A', 'literal-B']:
            j.begin(name, {'context_id': name}, self.read_state, self.read_counters)
            self.work(); self.prepare(j)
            before = list(self.captures); j.commit()
            self.assertEqual(self.captures, before)
        j.finish(self.read_state, self.read_counters)
        view = j.inspect()
        self.assertEqual(view['status'], 'complete')
        self.assertEqual(view['committed_ids'], ['literal-A', 'literal-B'])
        self.assertEqual(self.captures, [6, 6])
        self.assertEqual(view['terminal']['data']['observation']['state'], self.value)
        self.assertEqual(view['counters'], contract()['expected_counters'])
        self.refuse(lambda: j.begin('literal-A', {}, self.read_state, self.read_counters))

    def test_02_payload_by_value_and_observed_hash(self):
        j = self.create(); context = {'pair': ['raw', 'norm']}
        j.begin('literal-A', context, self.read_state, self.read_counters)
        context['pair'].append('later mutation'); self.work()
        result = {'literal_metric': [1, 2]}
        j.prepare(result, {'native': True, 'restored': True}, {'literal.weight': 2},
                  self.read_state, self.read_counters)
        result['literal_metric'].append(99)
        pending = j.inspect()['pending']
        self.assertEqual(pending['intent']['context'], {'pair': ['raw', 'norm']})
        self.assertEqual(pending['prepared']['result'], {'literal_metric': [1, 2]})
        self.assertEqual(pending['prepared']['state_witness']['actual_state_sha256'], digest(self.value))
        self.assertNotIn('state', pending['prepared'])

    def test_03_prepared_process_loss_commit_without_callbacks(self):
        self.child('prepared')
        j = Journal.recover(self.root); self.addCleanup(j.close)
        self.assertEqual(j.inspect()['status'], 'prepared')
        j.commit()
        self.assertEqual(j.inspect()['committed_ids'], ['literal-A'])
        self.assertEqual(j.inspect()['status'], 'open')
        self.refuse(lambda: j.begin('literal-B', {}, self.read_state, self.read_counters))
        self.refuse(lambda: j.finish(self.read_state, self.read_counters))
        self.assertEqual(self.captures, [0, 0])
        j.stop('Original process lost; final observation unavailable')
        self.assertEqual(j.inspect()['status'], 'incomplete')

    def test_04_intent_process_loss_remains_uncertain(self):
        self.child('intent')
        j = Journal.recover(self.root); self.addCleanup(j.close)
        self.assertEqual(j.inspect()['status'], 'uncertain_intent')
        self.refuse(j.commit)
        self.refuse(lambda: self.prepare(j))
        self.refuse(lambda: j.finish(self.read_state, self.read_counters))
        j.stop('Lost process with intent only')
        observed = j.inspect()['terminal']['data']['observation']
        self.assertIsNone(observed['state'])
        self.assertEqual(observed['errors']['state'], 'Unavailable')

    def test_05_process_exit_rolls_back_uncommitted_sql_transaction(self):
        self.child('uncommitted')
        j = Journal.recover(self.root); self.addCleanup(j.close)
        self.assertEqual(len(j.inspect()['events']), 1)
        self.assertEqual(j.inspect()['status'], 'uncertain_intent')
        self.refuse(j.commit)

    def test_06_lock_refusal_preserves_existing_evidence(self):
        j = self.create(); other = Journal.recover(self.root); self.addCleanup(other.close)
        j.con.execute('BEGIN IMMEDIATE')
        try:
            self.refuse(lambda: other.stop('Concurrent closure'), sqlite3.OperationalError)
        finally:
            j.con.execute('ROLLBACK')
        self.assertEqual(j.inspect()['events'], [])

    def test_07_stop_keeps_mismatch_and_callback_unavailability(self):
        j = self.create(); j.begin('literal-A', {}, self.read_state, self.read_counters)
        self.value['rng'] = 'changed-literal-rng'
        j.stop('Resource boundary', self.read_state, self.read_counters)
        self.assertEqual(j.inspect()['terminal']['data']['observation']['state']['rng'], 'changed-literal-rng')
        self.assertEqual(j.inspect()['status'], 'incomplete')
        self.refuse(j.commit)
        root = Path(self.tmp.name) / 'unavailable'
        k = Journal.create(root, contract(), state, counters); self.addCleanup(k.close)
        def unavailable():
            raise RuntimeError('Do not serialize exception messages')
        k.stop('Observation failed', unavailable, counters)
        observed = k.inspect()['terminal']['data']['observation']
        self.assertIsNone(observed['state']); self.assertEqual(observed['errors']['state'], 'RuntimeError')
        self.assertNotIn('Do not serialize', canonical(observed))

    def test_08_order_guard_state_and_counter_refusals(self):
        j = self.create()
        self.refuse(lambda: j.begin('literal-B', {}, self.read_state, self.read_counters))
        self.refuse(lambda: j.begin('unknown', {}, self.read_state, self.read_counters))
        j.begin('literal-A', {}, self.read_state, self.read_counters)
        self.refuse(lambda: j.begin('literal-A', {}, self.read_state, self.read_counters))
        self.refuse(j.commit); self.work()
        for checks in [{}, {'native': True, 'restored': False}, {'native': 1, 'restored': True}]:
            self.refuse(lambda: j.prepare({}, checks, {'literal.weight': 2}, self.read_state, self.read_counters))
        self.refuse(lambda: j.prepare({}, {'native': True, 'restored': True},
                                     {'unknown': 2}, self.read_state, self.read_counters))
        self.refuse(lambda: j.prepare({}, {'native': True, 'restored': True},
                                     {'literal.weight': -2}, self.read_state, self.read_counters))
        self.value['buffers']['literal'] = 'c' * 64
        self.refuse(lambda: self.prepare(j)); self.value['buffers']['literal'] = 'a' * 64
        for key, value in [('optimizer_steps', 1), ('forward_calls', True),
                           ('manual_parameter_assignments', 3), ('forward_calls', 7)]:
            saved = self.counts[key]; self.counts[key] = value
            self.refuse(lambda: self.prepare(j)); self.counts[key] = saved
        self.prepare(j); self.refuse(lambda: self.prepare(j)); j.commit()
        self.refuse(j.commit)
        self.refuse(lambda: j.finish(self.read_state, self.read_counters))
        self.assertEqual(j.inspect()['committed_ids'], ['literal-A'])

    def test_09_contract_json_and_overwrite_refusals(self):
        for change in [{'protocol_sha256': 'not-a-hash'}, {'measurement_ids': ['x', 'x']},
                       {'required_guards': []}, {'extra': True}]:
            spec = contract(); spec.update(change)
            self.refuse(lambda: Journal.create(self.root, spec, self.read_state, self.read_counters))
        self.assertEqual(self.captures, [0, 0])
        j = self.create(); before = list(self.captures)
        self.refuse(lambda: Journal.create(self.root, contract(), self.read_state, self.read_counters))
        self.assertEqual(self.captures, before)
        for value in [float('nan'), float('inf'), {'x': object()}, {1: 'key'}, {'x': 'z' * 2097153}]:
            self.refuse(lambda: canonical(value))
        self.refuse(lambda: j.begin('literal-A', [], self.read_state, self.read_counters))

    def test_10_append_only_and_content_tamper(self):
        j = self.create(); j.begin('literal-A', {}, self.read_state, self.read_counters)
        self.refuse(lambda: j.con.execute('DELETE FROM events'), sqlite3.IntegrityError)
        self.refuse(lambda: j.con.execute("UPDATE header SET body='{}'"), sqlite3.IntegrityError)
        j.con.execute('DROP TRIGGER no_events_UPDATE')
        j.con.execute("UPDATE events SET body=replace(body, 'literal-A', 'literal-Z')")
        self.refuse(j.inspect)

    def test_11_missing_event_and_final_counter_refusal(self):
        j = self.create()
        for name in ['literal-A', 'literal-B']:
            j.begin(name, {}, self.read_state, self.read_counters)
            self.work(); self.prepare(j); j.commit()
        saved = self.counts['input_token_exposures']; self.counts['input_token_exposures'] += 1
        self.refuse(lambda: j.finish(self.read_state, self.read_counters))
        self.counts['input_token_exposures'] = saved
        j.finish(self.read_state, self.read_counters)
        j.con.execute('DROP TRIGGER no_events_DELETE'); j.con.execute('DELETE FROM events WHERE seq=2')
        self.refuse(j.inspect)


if __name__ == '__main__':
    unittest.main()
