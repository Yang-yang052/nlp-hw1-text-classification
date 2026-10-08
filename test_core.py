"""Small checks of data isolation, representation and evaluation semantics."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from experiment import tokenize, average, metrics, prepare


def main():
    v = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    counts = v.fit_transform(['apple apple banana']).toarray()[0]
    assert dict(zip(v.get_feature_names_out(), counts)) == {'apple':2,'banana':1}
    v.binary = True
    assert v.transform(['apple apple banana unseen']).toarray().tolist() == [[1,1]]
    assert 'unseen' not in v.vocabulary_
    vec, stats = average([['a','a','b','unknown'], ['unknown']],
                         {'a':np.ones(100),'b':np.zeros(100)})
    assert np.allclose(vec[0], 2/3) and np.allclose(vec[1], 0)
    assert stats['empty_documents'] == 1 and stats['token_coverage'] == 3/5
    assert np.isclose(metrics([0,0], [0,0], [0,1,2])['macro_f1'], 1/3)
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='tmp') as folder:
        root = Path(folder)
        pd.DataFrame({'text':[f'news {i}' for i in range(100)],
                      'label':['a','b']*50}).to_csv(root/'nyt.csv',index=False)
        args = SimpleNamespace(nyt=root/'nyt.csv',seed=42,out=root)
        texts, _, _ = prepare(args)
        repeated, _, _ = prepare(args)
        assert texts == repeated and [len(t) for t in texts] == [80,10,10]
        assert not(set(texts[0]) & set(texts[1]) or set(texts[0]) & set(texts[2]))
        assert root.resolve().is_relative_to(Path.cwd().resolve())
    print('Core checks passed; synthetic data is not used for assignment results.')


if __name__ == '__main__':
    main()
