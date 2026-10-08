"""Download the assignment's official GloVe archive and extract only 100d."""
from pathlib import Path
import urllib.request
import zipfile

root = Path('assets')
root.mkdir(exist_ok=True)
target = root / 'glove.6B.100d.txt'
if not target.exists():
    archive = root / 'glove.6B.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        print('Downloading official glove.6B archive', flush=True)
        urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        z.extract('glove.6B.100d.txt', root)
print(target.resolve())
