"""Package source, data, measured results and the reviewed PDF; omit model caches."""
import hashlib
import json
import runpy
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
if Path.cwd().resolve() != root:
    raise SystemExit('Run this script from HW-1')
runpy.run_path(str(root/'validate_results.py'),run_name='__main__')
report = root/'实验报告.pdf'
if not report.exists() or report.stat().st_size == 0:
    raise SystemExit('Generate and visually review 实验报告.pdf before packaging')
names = ['experiment.py','report.py','analyze_results.py','test_core.py','validate_results.py','download_glove.py',
         'download_resumable.py','package_submission.py','run_all.ps1','README.md','requirements.txt',
         '.gitignore','.gitattributes','nyt.csv','ag.csv','实验报告.pdf','运行说明.md']
files = [root/n for n in names]
files += sorted((root/'results').glob('*.json'))
files += sorted((root/'results').glob('*.csv'))
manifest = {p.relative_to(root).as_posix(): {'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest_path = root/'提交清单.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
files.append(manifest_path)
archive = root.parent/'作业一提交包.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in files:
        z.write(path,'HW-1/'+path.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print('Verified submission archive:',archive,'files:',len(files),'bytes:',archive.stat().st_size)
