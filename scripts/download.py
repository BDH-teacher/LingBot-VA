"""Download pinned HF files and verify sizes. Never fetch demonstrations implicitly."""
import argparse, json, os, pathlib, shutil, zipfile
from huggingface_hub import snapshot_download
ROOT=pathlib.Path(os.environ['LINGBOT_UR5_ROOT'])
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['models','assets'])
    args=parser.parse_args()
    lock=json.loads((ROOT/'configs/sources.lock.json').read_text())['huggingface']
    for repo, spec in lock.items():
        if (args.kind=='assets') != (spec['type']=='dataset'): continue
        dest=ROOT/('third_party/RoboTwin-lingbot/assets' if args.kind=='assets' else 'checkpoints/pretrained/'+repo.split('/')[-1])
        missing=sum(x['size'] for x in spec['files'] if not (dest/x['path']).exists())
        if shutil.disk_usage(ROOT).free < missing + 20*1024**3:
            raise RuntimeError(f'Insufficient disk for {repo}: {missing} bytes plus 20GiB reserve')
        print(f'Downloading {repo}@{spec["revision"]} to {dest}',flush=True)
        snapshot_download(repo_id=repo, repo_type=spec['type'], revision=spec['revision'],
                          allow_patterns=[f['path'] for f in spec['files']], local_dir=dest, max_workers=2)
        for f in spec['files']:
            assert (dest/f['path']).stat().st_size==f['size'],f['path']
        (dest/'download_verified.json').write_text(json.dumps({'repo':repo,'revision':spec['revision'],'verified_sizes':True},indent=2))
        if args.kind=='assets':
            for f in spec['files']:
                z=dest/f['path']; stamp=dest/(z.stem+'.extracted')
                if not stamp.exists():
                    with zipfile.ZipFile(z) as archive:
                        for member in archive.infolist():
                            target=(dest/member.filename).resolve()
                            if not target.is_relative_to(dest.resolve()): raise ValueError('Unsafe zip member')
                        archive.extractall(dest)
                    stamp.write_text(spec['revision'])
        print('DOWNLOAD VERIFIED',repo,flush=True)
if __name__=='__main__': main()
