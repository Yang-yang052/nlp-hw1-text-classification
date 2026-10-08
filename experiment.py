"""Six comparable NYT classification experiments. No test-set model selection."""
import argparse
import importlib.metadata
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(y, pred, labels):
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)))


def prepare(args):
    frame = pd.read_csv(args.nyt)
    if not {'text', 'label'} <= set(frame.columns):
        raise ValueError('NYT CSV must contain text and label columns')
    if frame[['text', 'label']].isna().any().any():
        raise ValueError('Missing text or label; resolve before splitting')
    frame['text'] = frame.text.astype(str)
    frame['label'] = frame.label.astype(str)
    if (frame.text.str.strip() == '').any():
        raise ValueError('Empty documents found')
    names = sorted(frame.label.unique().tolist())
    y = frame.label.map({name: i for i, name in enumerate(names)}).to_numpy()
    order = np.random.default_rng(args.seed).permutation(len(frame))
    a, b = int(.8 * len(frame)), int(.9 * len(frame))
    idx = [order[:a], order[a:b], order[b:]]
    if set(y[idx[0]]) != set(range(len(names))):
        raise ValueError('Training split lacks a class; choose another seed')
    fingerprint = hashlib.sha256(Path(args.nyt).read_bytes()).hexdigest()
    save(args.out / 'split.json', dict(seed=args.seed, sha256=fingerprint,
         train=idx[0].tolist(), validation=idx[1].tolist(), test=idx[2].tolist(), labels=names))
    save(args.out / 'dataset.json', dict(n=len(frame), sizes=[len(i) for i in idx],
         counts=frame.label.value_counts().to_dict(), duplicate_texts=int(frame.text.duplicated().sum()),
         split_class_counts=[{names[c]: int((y[i] == c).sum()) for c in range(len(names))} for i in idx],
         overlapping_unique_texts={f'{i}_{j}':len(set(frame.text.iloc[idx[i]]) & set(frame.text.iloc[idx[j]]))
                                  for i,j in [(0,1),(0,2),(1,2)]}))
    return [frame.text.iloc[i].tolist() for i in idx], [y[i] for i in idx], names


def evaluate(name, pred, ys, names, args, extra):
    result = dict(method=name, test=metrics(ys[2], pred, list(range(len(names)))), **extra)
    result['classification_report'] = classification_report(ys[2], pred, labels=list(range(len(names))),
        target_names=names, output_dict=True, zero_division=0)
    result['confusion_matrix'] = confusion_matrix(ys[2], pred, labels=list(range(len(names)))).tolist()
    save(args.out / (name + '.json'), result)
    pd.DataFrame({'true': [names[i] for i in ys[2]], 'predicted': [names[i] for i in pred]}).to_csv(
        args.out / (name + '_predictions.csv'), index=False)
    print(name, result['test'], flush=True)


def logistic(name, xs, ys, names, args, scale=False, extra=None):
    best, chosen, score = None, None, -1
    trials = []
    for c in [.1, 1., 10.]:
        model = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=args.seed)
        if scale:
            model = make_pipeline(StandardScaler(), model)
        model.fit(xs[0], ys[0])
        validation = metrics(ys[1], model.predict(xs[1]), list(range(len(names))))
        trials.append(dict(C=c, **validation))
        if validation['macro_f1'] > score:
            best, chosen, score = model, c, validation['macro_f1']
    evaluate(name, best.predict(xs[2]), ys, names, args,
             dict(best_C=chosen, validation_trials=trials, **(extra or {})))


def average(tokens, vectors):
    result = np.zeros((len(tokens), 100), dtype=np.float32)
    hits, total, empty = 0, 0, 0
    for i, words in enumerate(tokens):
        known = [vectors[w] for w in words if w in vectors]
        total += len(words)
        hits += len(known)
        if known:
            result[i] = np.mean(known, axis=0)
        else:
            empty += 1
    return result, dict(token_coverage=hits / max(total, 1), empty_documents=empty)


def classical(texts, ys, names, args):
    tokens = [[tokenize(t) for t in part] for part in texts]
    for binary, name in [(True, 'binary_bow'), (False, 'word_frequency')]:
        if args.task not in ['all', 'bow']:
            continue
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, binary=binary)
        xs = [vectorizer.fit_transform(texts[0])] + [vectorizer.transform(t) for t in texts[1:]]
        logistic(name, xs, ys, names, args, extra=dict(vocabulary_size=len(vectorizer.vocabulary_)))
    if args.task not in ['all', 'embeddings', 'glove']:
        return
    from gensim.models import Word2Vec
    sources = [('word2vec_nyt', tokens[0]), ('word2vec_ag', [tokenize(t) for t in pd.read_csv(args.ag).text.astype(str)])]
    for name, corpus in (sources if args.task != 'glove' else []):
        print('Training', name, flush=True)
        model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2,
            workers=1, sg=1, negative=5, epochs=5, seed=args.seed,
            hashfxn=lambda w: int(hashlib.md5(w.encode()).hexdigest()[:8], 16))
        model.wv.save(str(args.out / (name + '.kv')))
        features = [average(part, model.wv) for part in tokens]
        logistic(name, [f[0] for f in features], ys, names, args, scale=True,
            extra=dict(coverage=[f[1] for f in features], vector_size=100, embedding_epochs=5))
    if not args.glove or not Path(args.glove).exists():
        raise FileNotFoundError('Supply --glove path/to/glove.6B.100d.txt')
    wanted = {w for part in tokens for doc in part for w in doc}
    vectors = {}
    with open(args.glove, encoding='utf-8') as stream:
        for line in stream:
            word, _, rest = line.partition(' ')
            if word in wanted:
                values = np.fromstring(rest, sep=' ', dtype=np.float32)
                if len(values) != 100:
                    raise ValueError('GloVe must use 100 dimensions')
                vectors[word] = values
    features = [average(part, vectors) for part in tokens]
    logistic('glove', [f[0] for f in features], ys, names, args, scale=True,
        extra=dict(coverage=[f[1] for f in features], vector_size=100))


def bert(texts, ys, names, args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={n:i for i,n in enumerate(names)}).to(device)
    class Encoded(Dataset):
        def __init__(self, ts, labels):
            self.data = tokenizer(ts, truncation=True, padding='max_length', max_length=64)
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            return {**{k: torch.tensor(v[i]) for k,v in self.data.items()}, 'labels': torch.tensor(self.labels[i])}
    loaders = [DataLoader(Encoded(t, y), batch_size=args.batch_size, shuffle=i == 0)
               for i,(t,y) in enumerate(zip(texts, ys))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    def predict(loader):
        model.eval()
        pred = []
        with torch.no_grad():
            for batch in loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    pred.extend(model(**batch).logits.argmax(-1).cpu().tolist())
        return pred
    history, best = [], -1
    checkpoint = args.out / 'bert_best'
    for epoch in range(3):
        model.train()
        total = 0.
        for step, batch in enumerate(loaders[0]):
            optimizer.zero_grad()
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                print('BERT epoch', epoch+1, 'step', step, 'loss', loss.item(), flush=True)
        validation = metrics(ys[1], predict(loaders[1]), list(range(len(names))))
        history.append(dict(epoch=epoch+1, loss=total/len(loaders[0]), **validation))
        save(args.out / 'bert_history.json', history)
        if validation['macro_f1'] > best:
            best = validation['macro_f1']
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        print(history[-1], flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    evaluate('bert', predict(loaders[2]), ys, names, args,
        dict(history=history, epochs=3, max_length=64, device=str(device), learning_rate=2e-5,
             batch_size=args.batch_size, bf16=args.bf16, threads=args.threads))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyt', default='nyt.csv')
    parser.add_argument('--ag', default='ag.csv')
    parser.add_argument('--glove')
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', choices=['all','bow','embeddings','glove','bert'], default='all')
    parser.add_argument('--bert-model', default='google-bert/bert-base-uncased')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 autocast on supported CPU/GPU')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'environment.json', {name: importlib.metadata.version(name) for name in
         ['numpy', 'pandas', 'scikit-learn', 'scipy', 'gensim', 'torch', 'transformers', 'reportlab']})
    texts, ys, names = prepare(args)
    print('Split sizes', [len(t) for t in texts], 'classes', names, flush=True)
    started = time.time()
    if args.task != 'bert':
        classical(texts, ys, names, args)
    if args.task in ['all','bert']:
        bert(texts, ys, names, args)
    print('Elapsed seconds', time.time()-started, flush=True)


if __name__ == '__main__':
    main()
