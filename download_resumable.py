"""Retry bounded HTTP ranges, preserve completed chunks, verify assembled length."""
import argparse
import concurrent.futures
import time
from pathlib import Path
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--chunk-mb', type=int, default=2)
    a = p.parse_args()
    proxies = {'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
    for attempt in range(15):
        try:
            response = requests.get(a.url, headers={'Range':'bytes=0-0'}, proxies=proxies, timeout=30, stream=True)
            response.raise_for_status()
            break
        except Exception:
            if attempt == 14:
                raise
            time.sleep(2)
    if response.status_code != 206:
        raise RuntimeError('Server does not support ranges')
    size = int(response.headers['Content-Range'].split('/')[-1])
    response.close()
    folder = a.output.with_name(a.output.name+'.chunks')
    folder.mkdir(parents=True,exist_ok=True)
    chunk = a.chunk_mb * 1024 * 1024
    n = (size+chunk-1)//chunk
    def fetch(i):
        start, end = i*chunk, min(size-1,(i+1)*chunk-1)
        path = folder / str(i)
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for attempt in range(20):
            try:
                with requests.get(a.url, headers={'Range':f'bytes={start}-{end}'}, proxies=proxies,
                                  timeout=(20,30),stream=True) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range','').split('/')[0] != f'bytes {start}-{end}':
                        raise RuntimeError('Incorrect range response')
                    with path.open('wb') as f:
                        for block in r.iter_content(65536):
                            f.write(block)
                if path.stat().st_size != end-start+1:
                    raise RuntimeError('Incomplete chunk')
                print(f'Completed chunk {i+1}/{n}',flush=True)
                return
            except Exception as error:
                print(f'Retry chunk {i+1}: {type(error).__name__} {str(error)[:180]}',flush=True)
                time.sleep(min(attempt+1,5))
        raise RuntimeError(f'Failed chunk {i}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(fetch,range(n)))
    with a.output.open('wb') as f:
        for i in range(n):
            f.write((folder/str(i)).read_bytes())
    assert a.output.stat().st_size == size
    print('Complete',a.output,size,flush=True)


if __name__ == '__main__':
    main()
