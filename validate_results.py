"""Recompute all reported metrics from exported predictions before submission."""
import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

root = Path('results')
split = json.loads((root/'split.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path('nyt.csv').read_bytes()).hexdigest() == split['sha256']
sets = [set(split[k]) for k in ['train','validation','test']]
assert all(not (sets[i] & sets[j]) for i,j in [(0,1),(0,2),(1,2)])
assert set.union(*sets) == set(range(len(pd.read_csv('nyt.csv'))))
source = pd.read_csv('nyt.csv').iloc[split['test']].label.astype(str).tolist()
methods = ['binary_bow','word_frequency','glove','word2vec_ag','word2vec_nyt','bert']
summary = []
for name in methods:
    record = json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
    predictions = pd.read_csv(root/(name+'_predictions.csv'))
    assert predictions.true.tolist() == source, name+' test row mismatch'
    assert np.isclose(record['test']['accuracy'], accuracy_score(predictions.true,predictions.predicted))
    assert np.isclose(record['test']['macro_f1'], f1_score(predictions.true,predictions.predicted,
                     labels=split['labels'],average='macro',zero_division=0))
    assert np.array_equal(record['confusion_matrix'],confusion_matrix(predictions.true,predictions.predicted,
                          labels=split['labels']))
    print(name,'metrics verified')
    summary.append({'method':name,**record['test']})
bert = json.loads((root/'bert.json').read_text(encoding='utf-8'))
assert bert['epochs'] == 3 and bert['max_length'] == 64 and len(bert['history']) == 3
with (root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=['method','accuracy','macro_f1'])
    writer.writeheader()
    writer.writerows(summary)
print('All six results verified; BERT trained three epochs with max_length=64.')
