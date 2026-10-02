import unittest, tempfile, json, sqlite3, copy
from pathlib import Path
import app as c

class CoreTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'test.sqlite'
        self.config = json.loads((Path(__file__).resolve().parent / 'scenario.json').read_text(encoding='utf-8'))

    def rows(self):
        return [{'features': [i % 3], 'label': i % 2, 'observed_at': i, 'label_at': i + 2} for i in range(20)]

    def test_training_cutoff(self):
        self.assertEqual(c.TemporalRisk().fit(self.rows(), 10).training_count, 8)

    def test_holdout_leakage(self):
        m = c.TemporalRisk().fit(self.rows(), 10)
        with self.assertRaises(ValueError):
            m.evaluate(self.rows()[:2])

    def test_probability(self):
        self.assertTrue(0 < c.TemporalRisk().fit(self.rows(), 10).predict([1]) < 1)

    def test_shape(self):
        m = c.TemporalRisk().fit(self.rows(), 10)
        with self.assertRaises(ValueError):
            m.predict([1, 2])

    def test_one_class(self):
        with self.assertRaises(ValueError):
            c.TemporalRisk().fit([{'features': [1], 'label': 1, 'observed_at': 1, 'label_at': 2}], 10)

    def test_deterministic(self):
        self.assertEqual(c.run({'seed': 4}), c.run({'seed': 4}))

    def test_nonfinite_training(self):
        rows = self.rows()
        rows[0]['features'] = [float('nan')]
        with self.assertRaises(ValueError):
            c.TemporalRisk().fit(rows, 10)

    def test_future_label_excluded(self):
        rows = self.rows()
        rows[0]['label_at'] = 100
        self.assertEqual(c.TemporalRisk().fit(rows, 10).training_count, 7)
if __name__ == '__main__':
    unittest.main()
