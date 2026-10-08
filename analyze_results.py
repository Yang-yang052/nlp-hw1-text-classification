"""Derive additional report evidence without changing or retraining any model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from experiment import tokenize, metrics, save


def main():
    root = Path('results')
    frame = pd.read_csv('nyt.csv')
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    words = [len(tokenize(t)) for t in frame.text]
    tokenizer_path = Path('assets/bert-base-uncased/tokenizer.json')
    if not tokenizer_path.exists():
        tokenizer_path = Path('results/bert_best/tokenizer.json')
    if not tokenizer_path.exists():
        raise SystemExit('Additional analysis requires the original or saved BERT tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    encodings = tokenizer.encode_batch(frame.text.tolist())
    lengths = np.array([len(e.ids) for e in encodings])
    stats = dict(word_length=dict(mean=float(np.mean(words)),median=float(np.median(words)),
                 p95=float(np.percentile(words,95)),maximum=int(max(words))),
                 bert_length=dict(mean=float(np.mean(lengths)),median=float(np.median(lengths)),
                 p95=float(np.percentile(lengths,95))),
                 bert_truncation={k:dict(n=len(split[k]),truncated=int((lengths[split[k]]>64).sum()),
                     fraction=float((lengths[split[k]]>64).mean())) for k in ['train','validation','test']})
    methods = ['binary_bow','word_frequency','glove','word2vec_ag','word2vec_nyt','bert']
    truth = frame.label.iloc[split['test']].tolist()
    predictions = {n:pd.read_csv(root/(n+'_predictions.csv')).predicted.tolist() for n in methods}
    baseline = ['sports']*len(truth)
    stats['majority_baseline'] = metrics(truth,baseline,split['labels'])
    stats['error_counts'] = {n:sum(a!=b for a,b in zip(truth,predictions[n])) for n in methods}
    stats['paired_bert_bow'] = {
        'both_correct':sum(a==b==c for a,b,c in zip(truth,predictions['bert'],predictions['binary_bow'])),
        'bow_only_correct':sum(a==c and a!=b for a,b,c in zip(truth,predictions['bert'],predictions['binary_bow'])),
        'bert_only_correct':sum(a==b and a!=c for a,b,c in zip(truth,predictions['bert'],predictions['binary_bow'])),
        'both_wrong':sum(a!=b and a!=c for a,b,c in zip(truth,predictions['bert'],predictions['binary_bow']))}
    examples = []
    selected = set()
    # Select the first example of each misclassification direction; no cherry-picked success cases.
    for name in ['binary_bow','bert']:
        directions = set()
        for i,(true,pred) in enumerate(zip(truth,predictions[name])):
            if true==pred or (true,pred) in directions:
                continue
            directions.add((true,pred))
            row = split['test'][i]
            text = frame.text.iloc[row]
            examples.append(dict(method=name,row_id=row,test_position=i,true=true,predicted=pred,
                excerpt=text[:420],bert_prefix=tokenizer.decode(encodings[row].ids[:63]),
                wordpiece_length=int(lengths[row]),predictions={n:predictions[n][i] for n in methods}))
            selected.add(row)
    stats['examples'] = examples
    save(root/'analysis.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if k!='examples'},ensure_ascii=False,indent=2))
    for x in examples:
        print(x['method'],x['row_id'],x['true'],'->',x['predicted'],x['excerpt'],flush=True)


if __name__ == '__main__':
    main()
