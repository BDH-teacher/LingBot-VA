"""Export only public source files; never traverse local data or symlinks."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = {'src', 'configs', 'envs', 'patches', 'scripts', 'tests'}
PUBLIC_FILES = {'README.md', '.gitignore', 'pyproject.toml'}


def source_files(root=ROOT):
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        check=True, capture_output=True,
    )
    names = sorted(set(result.stdout.decode().split('\0')) - {''})
    selected = []
    for name in names:
        path = Path(name)
        if name not in PUBLIC_FILES and path.parts[0] not in PUBLIC_DIRS:
            continue
        # Tracked files can still match newly added ignore rules.
        ignored = subprocess.run(
            ['git', '-C', str(root), 'check-ignore', '--no-index', '-q', '--', name],
            check=False,
        )
        if ignored.returncode == 0:
            continue
        if ignored.returncode != 1:
            raise RuntimeError(f'Cannot check ignore rules: {name}')
        absolute = root / path
        if any(p.is_symlink() for p in [absolute, *absolute.parents] if p != root.parent):
            raise ValueError(f'Symlinks are not release inputs: {name}')
        if not absolute.is_file():
            raise ValueError(f'Missing release input: {name}')
        if absolute.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f'Unexpected large source file: {name}')
        selected.append(path)
    if not PUBLIC_FILES.issubset({p.as_posix() for p in selected}):
        raise ValueError('Required root files are missing')
    return selected


def main():
    paths = source_files()
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    archive = output / 'lingbot-ur5-source.zip'
    temporary = output / 'lingbot-ur5-source.tmp'
    manifest = []
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            content = (ROOT / path).read_bytes()
            bundle.write(ROOT / path, arcname=path.as_posix())
            manifest.append({'path': path.as_posix(), 'bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest()})
    temporary.replace(archive)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'SHA256SUMS').write_text(f'{checksum}  {archive.name}\n')
    print(f'{archive}: {len(paths)} files, {archive.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
