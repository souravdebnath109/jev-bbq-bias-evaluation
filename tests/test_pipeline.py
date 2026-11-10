import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jev_bbq


def row(id, context='ambig', label=2):
    return {'example_id': id, 'context': 'Maya is a doctor. Rahim is a nurse.' if context == 'disambig' else 'Maya and Rahim work at a hospital.',
            'question': 'Who is the doctor?', 'ans0': 'Maya', 'ans1': 'Rahim', 'ans2': 'Cannot be determined',
            'label': label, 'context_condition': context, 'answer_info': {'ans0': ['Maya', 'name'], 'ans1': ['Rahim', 'name'], 'ans2': ['unknown', 'unknown']},
            'category': 'Age', '_key': f'Age:{id}', '_category': 'Age'}


class PipelineTests(unittest.TestCase):
    def test_request_exactly_one_input(self):
        item = row(4, 'disambig', 0)
        req = jev_bbq.make_request(item)
        self.assertEqual(req['state'], item['context'])
        self.assertNotIn('label', json.dumps(req))
        self.assertEqual(list(req['questions']['bbq']['criteria']), ['A', 'B', 'C'])
        rev = jev_bbq.make_request(item, reverse=True)
        self.assertEqual(list(rev['questions']['bbq']['criteria']), ['C', 'B', 'A'])

    def test_balanced_sampling(self):
        rows = [row(i, 'ambig', 2) for i in range(20)] + [row(i+20, 'disambig', 0) for i in range(20)]
        selected = jev_bbq.select_sample(rows, 10, 2026)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(r['context_condition']=='ambig' for r in selected), 5)
        self.assertEqual(len({r['_key'] for r in selected}), 10)

    def test_predict_uses_real_response_schema(self):
        simulated = {'model':'jev-1.13-free', 'answers': {'bbq': {'type': 'choice', 'choice':'A',
                     'confidence': 1, 'probabilities': {'C':0, 'B':0, 'A':1}}},
                     'usage': {'input_tokens': 333, 'output_tokens': 38, 'cost':0}}
        with patch.object(jev_bbq, 'call_jev', return_value=simulated) as mock_call:
            result = jev_bbq.predict_one(row(3,'disambig',0), 'not-a-real-key')
        mock_call.assert_called_once()
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['probabilities'], [1., 0., 0.])
        self.assertEqual(result['response']['usage']['cost'], 0)
        self.assertNotIn('not-a-real-key', json.dumps(result))

    def test_score_ambig_and_disambig(self):
        dataset = [row(1, 'ambig', 2), row(2, 'disambig', 0), row(3, 'disambig', 0)]
        predictions = [{'key':'Age:1','prediction':2,'probabilities':[0.1,0.1,0.8]},
                       {'key':'Age:2','prediction':0,'probabilities':[0.9,0.05,0.05]},
                       {'key':'Age:3','prediction':2,'probabilities':[0.01,0.01,0.98]}]
        result, details = jev_bbq.summarize(predictions, dataset)
        self.assertEqual(result['overall']['correct'], 2)
        self.assertEqual(result['by_context']['disambig']['unnecessary_abstentions'], 1)
        self.assertEqual(result['by_context']['ambig']['unsupported_ambiguous_guesses'], 0)
        self.assertEqual(details[0]['unknown_option'], 2)

    def test_dataset_reader(self):
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d)/'Age.jsonl','w',encoding='utf8') as f:
                r=row(1)
                f.write(json.dumps({k:v for k,v in r.items() if not k.startswith('_')})+'\n')
            rows=jev_bbq.read_bbq(d)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['_key'],'Age:1')

if __name__ == '__main__':
    unittest.main()
