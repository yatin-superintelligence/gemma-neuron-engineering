"""Strict random control matching refuses small pools and preserves ranking."""
import unittest
import numpy as np
from methods.strict_profile_matching import select


class StrictProfileMatchingTests(unittest.TestCase):
    def data(self):
        d={k:np.ones(64,dtype=np.float32)for k in ['RMS','delta_RMS','down_norm']};d['min_score']=np.zeros(64,dtype=np.float32);d['min_score'][:4]=[4,3,2,1];return d
    def test_exact_ranking_and_deterministic_controls(self):
        d=self.data();a=select(d,[64],count=4);self.assertEqual(a,select(d,[64],count=4));self.assertEqual([c['flat']for c in a['groups']['selected4']],[0,1,2,3])
        for name,g in a['groups'].items():
            if name.startswith('random'):self.assertTrue(all(c['score']==0 for c in g));self.assertFalse({c['flat']for c in g}&{0,1,2,3})
    def test_small_pool_stops(self):
        a=select({k:v[:8]for k,v in self.data().items()},[8],count=4);self.assertEqual(a['status'],'planned_host_matching_stop');self.assertFalse(a['groups'])
    def test_unmatchable_top_coordinate_is_recorded(self):
        d=self.data();d['RMS'][0]=100;a=select(d,[64],count=2);self.assertEqual(a['excluded_candidates'][0]['rejected']['flat'],0);self.assertEqual([c['flat']for c in a['groups']['selected2']],[1,2])


if __name__=='__main__':unittest.main()
