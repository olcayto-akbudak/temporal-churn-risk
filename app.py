"""Zamansal Müşteri Risk Modeli.

Problem: Henüz bilinmeyen gelecek etiketlerinin eğitim setine sızmasını engelleyerek risk skorlamak.
Method: Lojistik regresyon, label olgunlaşması, holdout
Invariant: Etiket zamanı cutoff öncesinde olmalıdır; holdout gözlem zamanı cutoff sonrasındadır.
Boundary: Sentetik veri ve basit modeldir; gerçek müşteride grup ayrımı, kalibrasyon ve model yönetişimi gerekir."""
import math, random

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-40, min(40, x))))

class TemporalRisk:

    def fit(self, rows, cutoff, epochs=400):
        if epochs < 1:
            raise ValueError('positive epochs')
        if any((not r['features'] or any((not math.isfinite(v) for v in r['features'])) for r in rows)):
            raise ValueError('finite features required')
        train = [r for r in rows if r['observed_at'] < cutoff and r['label_at'] < cutoff]
        if not train or len({r['label'] for r in train}) < 2:
            raise ValueError('two historical classes required')
        self.width = len(train[0]['features'])
        if any((len(r['features']) != self.width or r['label_at'] < r['observed_at'] or r['label'] not in (0, 1) for r in rows)):
            raise ValueError('invalid temporal record')
        self.means = [sum((r['features'][j] for r in train)) / len(train) for j in range(self.width)]
        self.scales = [max(1e-08, (sum(((r['features'][j] - self.means[j]) ** 2 for r in train)) / len(train)) ** 0.5) for j in range(self.width)]
        self.weights = [0.0] * (self.width + 1)
        for _ in range(epochs):
            gradient = [0.0] * (self.width + 1)
            for r in train:
                x = [1] + [(v - m) / s for v, m, s in zip(r['features'], self.means, self.scales)]
                error = sigmoid(sum((w * v for w, v in zip(self.weights, x)))) - r['label']
                for j, v in enumerate(x):
                    gradient[j] += error * v / len(train)
            self.weights = [w - 0.08 * (g + (0.01 * w if j else 0)) for j, (w, g) in enumerate(zip(self.weights, gradient))]
        self.cutoff = cutoff
        self.training_count = len(train)
        return self

    def predict(self, features):
        if len(features) != self.width or any((not math.isfinite(v) for v in features)):
            raise ValueError('feature vector')
        x = [1] + [(v - m) / s for v, m, s in zip(features, self.means, self.scales)]
        return sigmoid(sum((w * v for w, v in zip(self.weights, x))))

    def evaluate(self, rows, threshold=0.5):
        if any((r['observed_at'] < self.cutoff for r in rows)):
            raise ValueError('holdout predates cutoff')
        if not rows or not 0 < threshold < 1 or any((r['label'] not in (0, 1) for r in rows)):
            raise ValueError('invalid holdout/threshold')
        probabilities = [self.predict(r['features']) for r in rows]
        matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for r, p in zip(rows, probabilities):
            matrix[('t' if (p >= threshold) == bool(r['label']) else 'f') + ('p' if p >= threshold else 'n')] += 1
        return {'confusion': matrix, 'brier': sum(((p - r['label']) ** 2 for r, p in zip(rows, probabilities))) / len(rows), 'cost': matrix['fn'] * 5 + matrix['fp'], 'scores': probabilities}

def run(config):
    rng = random.Random(config.get('seed', 42))
    rows = []
    for day in range(1, 201):
        usage = rng.random()
        tickets = rng.randrange(6)
        y = int(rng.random() < sigmoid(-2 - 3 * usage + 0.7 * tickets))
        rows.append({'observed_at': day, 'label_at': day + 7, 'features': [usage, tickets], 'label': y})
    model = TemporalRisk().fit(rows, 150)
    return {'training_count': model.training_count, 'cutoff': 150, 'holdout': model.evaluate([r for r in rows if r['observed_at'] >= 150]), 'weights': model.weights, 'scope': 'Synthetic logistic model; no real customer prediction.'}

import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run reproducible synthetic project scenario')
    parser.add_argument('command', choices=['demo'])
    parser.add_argument('--input', default='scenario.json')
    parser.add_argument('--output', default='report.json')
    args = parser.parse_args()
    report = run(json.loads(Path(args.input).read_text(encoding='utf-8')))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Report: {target}')
if __name__ == '__main__':
    main()
