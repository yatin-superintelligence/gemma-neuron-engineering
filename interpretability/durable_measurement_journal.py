"""Append-only observation receipts for future measurements, not model certification.

Call begin BEFORE external work, prepare AFTER restoration, then commit. A durable
prepared event can be committed after process loss without repeating external
work. An intent alone remains uncertain. This module never loads or executes a
model. Callers remain responsible for the origin and scope of their observations.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote
import uuid


COUNTERS = ('forward_calls', 'input_token_exposures', 'manual_parameter_assignments',
            'activation_assignments', 'new_generated_tokens', 'backward_calls', 'optimizer_steps')
ZERO = ('new_generated_tokens', 'backward_calls', 'optimizer_steps')
MAX_BYTES = 2 * 1024 * 1024


class JournalError(ValueError):
    pass


def require(value, message):
    if not value:
        raise JournalError(message)


def canonical(value):
    def check(item):
        if type(item) is dict:
            require(all(type(k) is str for k in item), 'JSON keys must be strings')
            for v in item.values():
                check(v)
        elif type(item) is list:
            for v in item:
                check(v)
        else:
            require(type(item) in (str, int, float, bool, type(None)), 'JSON-only observation required')
            require(type(item) is not float or math.isfinite(item), 'Nonfinite observation')
    check(value)
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(len(text.encode()) <= MAX_BYTES, 'Observation exceeds bounded receipt size')
    return text


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _counters(value):
    require(type(value) is dict and set(value) == set(COUNTERS), 'Counter schema mismatch')
    require(all(type(v) is int and v >= 0 for v in value.values()), 'Nonnegative integer counters required')
    require(all(value[k] == 0 for k in ZERO), 'Generation, backwards or optimizer updates are outside scope')


def _state(value):
    require(type(value) is dict and type(value.get('parameters')) is dict
            and bool(value['parameters']), 'Parameter metadata snapshot required')
    for name, item in value['parameters'].items():
        require(type(name) is str and type(item) is dict and type(item.get('version')) is int
                and item['version'] >= 0, 'Invalid observed parameter version')
    canonical(value)


def _expected(baseline, deltas):
    result = copy.deepcopy(baseline)
    for name, increment in deltas.items():
        require(name in result['parameters'] and type(increment) is int and increment > 0,
                'Unknown or nonpositive declared version increment')
        result['parameters'][name]['version'] += increment
    return result


def _observation(state_reader, counter_reader):
    result = {'state': None, 'counters': None, 'errors': {}, 'pid': os.getpid(),
              'time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    for key, reader in [('state', state_reader), ('counters', counter_reader)]:
        try:
            # Serialize immediately: later caller mutation cannot change the receipt.
            result[key] = json.loads(canonical(reader()))
        except Exception as error:
            result['errors'][key] = type(error).__name__
    return result


def _contract(value):
    require(type(value) is dict and set(value) == {'protocol_sha256', 'source_sha256',
            'measurement_ids', 'required_guards', 'expected_counters'}, 'Contract schema mismatch')
    require(_hash(value['protocol_sha256']) and _hash(value['source_sha256']), 'Pinned source hashes required')
    for key in ['measurement_ids', 'required_guards']:
        items = value[key]
        require(type(items) is list and bool(items) and all(type(x) is str and 0 < len(x) <= 160
                for x in items) and len(set(items)) == len(items), 'Ordered unique IDs/guards required')
    _counters(value['expected_counters'])
    canonical(value)


def _connect(root):
    path = (Path(root) / 'journal.sqlite3').resolve()
    require(path.is_file() and not (Path(root) / 'journal.sqlite3').is_symlink(), 'Existing owned journal required')
    con = sqlite3.connect('file:' + quote(str(path), safe='/') + '?mode=rw', uri=True,
                          timeout=0, isolation_level=None)
    con.execute('PRAGMA synchronous=FULL')
    require(con.execute('PRAGMA journal_mode').fetchone()[0] == 'delete', 'Rollback journal mode required')
    return con


def _scan(con):
    rows = con.execute('SELECT body, digest FROM header').fetchall()
    require(len(rows) == 1, 'One immutable header required')
    header = json.loads(rows[0][0])
    require(digest(header) == rows[0][1], 'Header digest mismatch')
    require(header['schema'] == 1, 'Unknown journal schema')
    _contract(header['contract'])
    base = header['baseline']
    require(not base['errors'], 'Baseline observation unavailable')
    _state(base['state']); _counters(base['counters'])
    view = {'header': header, 'head': rows[0][1], 'events': [], 'committed_ids': [],
            'pending': None, 'terminal': None, 'deltas': {}, 'counters': base['counters']}
    for seq, body, saved_hash in con.execute('SELECT seq, body, digest FROM events ORDER BY seq'):
        event = json.loads(body)
        require(seq == len(view['events']) + 1 and event['seq'] == seq, 'Missing or reordered event')
        require(event['previous'] == view['head'] and digest(event) == saved_hash, 'Event digest chain mismatch')
        require(view['terminal'] is None, 'Event after terminal closure')
        kind, data = event['kind'], event['data']
        ids = header['contract']['measurement_ids']
        if kind == 'intent':
            require(view['pending'] is None and len(view['committed_ids']) < len(ids)
                    and data['id'] == ids[len(view['committed_ids'])], 'Measurement order/duplicate mismatch')
            require(data['before']['state_sha256'] == digest(_expected(base['state'], view['deltas']))
                    and data['before']['counters'] == view['counters'], 'Intent observation mismatch')
            view['pending'] = {'intent': data, 'prepared': None, 'prepared_digest': None}
        elif kind == 'prepared':
            pending = view['pending']
            require(pending is not None and pending['prepared'] is None
                    and data['id'] == pending['intent']['id'], 'Prepared measurement mismatch')
            checks = data['checks']
            require(type(checks) is dict and set(checks) == set(header['contract']['required_guards'])
                    and all(v is True for v in checks.values()), 'Missing or failed guard')
            require(type(data['version_deltas']) is dict, 'Declared version mapping required')
            new = dict(view['deltas'])
            for name, count in data['version_deltas'].items():
                require(type(count) is int and count > 0, 'Invalid version increment')
                new[name] = new.get(name, 0) + count
            expected = _expected(base['state'], new)
            witness = data['state_witness']
            require(witness['actual_state_sha256'] == digest(expected)
                    and witness['changed_parameter_versions'] == {n: expected['parameters'][n]['version'] for n in new},
                    'Prepared observed state witness mismatch')
            counters = data['counters']; _counters(counters)
            require(all(counters[k] >= view['counters'][k] for k in COUNTERS), 'Counter decrease')
            require(all(counters[k] <= header['contract']['expected_counters'][k] for k in COUNTERS),
                    'Counters exceed fixed declared work')
            require(counters['manual_parameter_assignments'] - base['counters']['manual_parameter_assignments']
                    == sum(new.values()), 'Parameter assignment/version mismatch')
            pending.update(prepared=data, prepared_digest=saved_hash)
        elif kind == 'committed':
            pending = view['pending']
            require(pending is not None and pending['prepared'] is not None
                    and data == {'id': pending['intent']['id'], 'prepared_digest': pending['prepared_digest']},
                    'Committed reference mismatch')
            prepared = pending['prepared']
            for name, count in prepared['version_deltas'].items():
                view['deltas'][name] = view['deltas'].get(name, 0) + count
            view['counters'] = prepared['counters']
            view['committed_ids'].append(data['id']); view['pending'] = None
        elif kind in ('complete', 'incomplete'):
            observation = data['observation']
            if kind == 'complete':
                require(view['pending'] is None and view['committed_ids'] == ids, 'Incomplete fixed cohort')
                require(not observation['errors'] and observation['state'] == _expected(base['state'], view['deltas'])
                        and observation['counters'] == header['contract']['expected_counters']
                        == view['counters'], 'Final observed state/counters mismatch')
            view['terminal'] = event
        else:
            raise JournalError('Unknown event kind')
        view['events'].append(event); view['head'] = saved_hash
    view['status'] = ('complete' if view['terminal'] and view['terminal']['kind'] == 'complete' else
                      'incomplete' if view['terminal'] else 'prepared' if view['pending'] and
                      view['pending']['prepared'] else 'uncertain_intent' if view['pending'] else 'open')
    return view


class Journal:
    """One creating process writes observations; recovery only commits saved evidence."""
    @classmethod
    def create(cls, root, contract, state_reader, counter_reader):
        contract = json.loads(canonical(contract)); _contract(contract)
        root = Path(root)
        require(not root.exists() and not root.is_symlink(), 'Journal overwrite refused')
        baseline = _observation(state_reader, counter_reader)
        require(not baseline['errors'], 'Actual baseline observation required')
        _state(baseline['state']); _counters(baseline['counters'])
        require(all(baseline['counters'][k] <= contract['expected_counters'][k] for k in COUNTERS),
                'Baseline exceeds declared final counters')
        root.mkdir(exist_ok=False)
        con = sqlite3.connect(root / 'journal.sqlite3', timeout=0, isolation_level=None)
        con.execute('PRAGMA journal_mode=DELETE'); con.execute('PRAGMA synchronous=FULL')
        obj = cls.__new__(cls); obj.con = con; obj.session = uuid.uuid4().hex; obj.pid = os.getpid()
        header = {'schema': 1, 'contract': contract, 'baseline': baseline,
                  'creator_session': obj.session, 'creator_pid': obj.pid}
        try:
            con.execute('BEGIN IMMEDIATE')
            con.execute('CREATE TABLE header (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL, digest TEXT NOT NULL)')
            con.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL UNIQUE)')
            for table in ['header', 'events']:
                for action in ['UPDATE', 'DELETE']:
                    con.execute(f"CREATE TRIGGER no_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'Append only'); END")
            con.execute('INSERT INTO header VALUES (1, ?, ?)', (canonical(header), digest(header)))
            con.execute('COMMIT')
        except BaseException:
            if con.in_transaction: con.execute('ROLLBACK')
            con.close(); raise
        return obj

    @classmethod
    def recover(cls, root):
        obj = cls.__new__(cls); obj.con = _connect(root); obj.session = None; obj.pid = os.getpid()
        try: _scan(obj.con)
        except BaseException: obj.con.close(); raise
        return obj

    def inspect(self):
        return _scan(self.con)

    def close(self):
        self.con.close()

    def _append(self, kind, maker):
        try:
            self.con.execute('BEGIN IMMEDIATE')
            view = _scan(self.con)
            require(view['terminal'] is None, 'Journal already terminal')
            data = maker(view)
            event = {'seq': len(view['events']) + 1, 'previous': view['head'],
                     'kind': kind, 'data': data, 'writer_pid': os.getpid(), 'time_ns': time.time_ns()}
            self.con.execute('INSERT INTO events VALUES (?, ?, ?)',
                             (event['seq'], canonical(event), digest(event)))
            _scan(self.con)  # Validate the transaction before making it durable.
            self.con.execute('COMMIT')
            return digest(event)
        except BaseException:
            if self.con.in_transaction: self.con.execute('ROLLBACK')
            raise

    def _creator(self, view):
        require(self.session == view['header']['creator_session'] and os.getpid() == self.pid
                == view['header']['creator_pid'], 'Original observation process required')

    def begin(self, measurement_id, context, state_reader, counter_reader):
        context = json.loads(canonical(context))
        require(type(context) is dict, 'Context mapping required')
        def make(view):
            self._creator(view)
            require(view['pending'] is None, 'Pending work cannot be retried or skipped')
            ids = view['header']['contract']['measurement_ids']
            require(len(view['committed_ids']) < len(ids) and measurement_id == ids[len(view['committed_ids'])],
                    'Wrong, duplicate or reordered measurement ID')
            observed = _observation(state_reader, counter_reader)
            require(not observed['errors'], 'Actual starting observation unavailable')
            _state(observed['state']); _counters(observed['counters'])
            return {'id': measurement_id, 'context': context,
                    'before': {'state_sha256': digest(observed['state']), 'counters': observed['counters'],
                               'capture_time_ns': observed['time_ns']}}
        return self._append('intent', make)

    def prepare(self, result, checks, version_deltas, state_reader, counter_reader):
        result, checks, version_deltas = [json.loads(canonical(v)) for v in [result, checks, version_deltas]]
        require(type(result) is dict and type(checks) is dict and type(version_deltas) is dict,
                'Result, guards and versions require mappings')
        def make(view):
            self._creator(view)
            require(view['pending'] is not None and view['pending']['prepared'] is None, 'One pending intent required')
            observed = _observation(state_reader, counter_reader)
            require(not observed['errors'], 'Actual completion observation unavailable; stop as incomplete')
            _state(observed['state']); _counters(observed['counters'])
            new = dict(view['deltas'])
            for name, count in version_deltas.items():
                require(type(count) is int and count > 0, 'Invalid declared version delta')
                new[name] = new.get(name, 0) + count
            expected = _expected(view['header']['baseline']['state'], new)
            require(observed['state'] == expected, 'Observed state has not restored')
            return {'id': view['pending']['intent']['id'], 'result': result, 'checks': checks,
                    'version_deltas': version_deltas, 'counters': observed['counters'],
                    'state_witness': {'actual_state_sha256': digest(observed['state']),
                        'changed_parameter_versions': {n: observed['state']['parameters'][n]['version'] for n in new},
                        'capture_time_ns': observed['time_ns']}}
        return self._append('prepared', make)

    def commit(self):
        def make(view):
            pending = view['pending']
            require(pending is not None and pending['prepared'] is not None, 'Durable prepared evidence required')
            return {'id': pending['intent']['id'], 'prepared_digest': pending['prepared_digest']}
        return self._append('committed', make)

    def finish(self, state_reader, counter_reader):
        def make(view):
            self._creator(view)
            return {'observation': _observation(state_reader, counter_reader)}
        return self._append('complete', make)

    def stop(self, reason, state_reader=None, counter_reader=None):
        require(type(reason) is str and 0 < len(reason) <= 500, 'Bounded interruption reason required')
        def make(view):
            if state_reader is None and counter_reader is None:
                observation = {'state': None, 'counters': None,
                               'errors': {'state': 'Unavailable', 'counters': 'Unavailable'},
                               'pid': None, 'time_ns': time.time_ns()}
            else:
                self._creator(view)
                require(state_reader is not None and counter_reader is not None, 'Both observation callbacks required')
                observation = _observation(state_reader, counter_reader)
            return {'reason': reason, 'observation': observation,
                    'original_final_state_not_certified': True}
        return self._append('incomplete', make)
