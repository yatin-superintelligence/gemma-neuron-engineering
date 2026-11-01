"""Lossless bounded private output transport with strict member provenance."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(data): return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or str(path) != name:
        raise ValueError('Strict relative output member required')


def pack(source, archive, manifest_path, maximum=67108864):
    source, archive, manifest_path = map(Path, (source, archive, manifest_path))
    if archive.exists() or manifest_path.exists(): raise ValueError('Never overwrite transport evidence')
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if not files or any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Actual regular output files required')
    records = {}
    total = 0
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in files:
            name = path.relative_to(source).as_posix(); safe_name(name)
            data = path.read_bytes(); total += len(data)
            if total > maximum: raise ValueError('Raw output storage boundary exceeded')
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = (stat.S_IFREG | 0o600) << 16
            stream.writestr(info, data)
            records[name] = {'bytes': len(data), 'sha256': sha(data)}
    manifest = {'schema': 'digital-surgery-output-zip-v1', 'raw_bytes': total,
                'archive_sha256': sha(archive.read_bytes()), 'files': records}
    verify(archive, manifest, maximum)
    with manifest_path.open('x') as stream: stream.write(json.dumps(manifest, indent=2)+'\n')
    return manifest


def verify(archive, manifest, maximum=67108864):
    if manifest.get('schema') != 'digital-surgery-output-zip-v1' or sha(Path(archive).read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Exact archive identity required')
    if type(manifest['raw_bytes']) is not int or not 0 <= manifest['raw_bytes'] <= maximum: raise ValueError('Raw storage boundary invalid')
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']): raise ValueError('Exact unique member inventory required')
        total = 0
        for info in stream.infolist():
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16): raise ValueError('Only regular file members accepted')
            record = manifest['files'][info.filename]
            if type(record['bytes']) is not int or record['bytes'] < 0 or info.file_size != record['bytes']: raise ValueError('Strict actual member size required')
            total += info.file_size
            if total > maximum: raise ValueError('Expanded output boundary exceeded')
            data = stream.read(info)
            if len(data) != record['bytes'] or sha(data) != record['sha256']: raise ValueError('Member byte identity differs')
        if total != manifest['raw_bytes']: raise ValueError('Expanded total differs')


def unpack(archive, manifest, destination, maximum=67108864):
    destination = Path(destination)
    if destination.exists(): raise ValueError('Never overwrite received raw evidence')
    verify(archive, manifest, maximum)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as stream:
        for name in stream.namelist():
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()): raise ValueError('Output path escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output: output.write(stream.read(name))
