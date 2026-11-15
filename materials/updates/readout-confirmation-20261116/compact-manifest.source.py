"""Preserve the exact full manifest as lossless gzip in Git; native audited run is unchanged."""
from pathlib import Path
import gzip, hashlib, json

ROOT=Path('/home/husrcf/Code/SAS')
BASE=Path('/media/990Pro/Code/SAS')
DEST=ROOT/'materials/updates/readout-confirmation-20261116'
source=BASE/'runs/memory_update/readout-confirmation-primary-20261116-01/manifest.json'
copy=DEST/'primary-run-manifest.json'
out=DEST/'primary-run-manifest.json.gz'
assert not out.exists()
original=source.read_bytes()
assert copy.read_bytes()==original
packed=gzip.compress(original,compresslevel=6,mtime=0)
out.write_bytes(packed)
assert gzip.decompress(out.read_bytes())==original
sha=lambda data:hashlib.sha256(data).hexdigest()
receipt={'status':'original_manifest_exact_bytes_preserved_in_lossless_gzip',
         'audited_native_run_manifest':str(source),'original_sha256':sha(original),
         'original_bytes':len(original),'gzip_sha256':sha(packed),'gzip_bytes':len(packed),
         'audited_native_run_unchanged':True,'scientific_source_files_changed':False}
(DEST/'primary-run-manifest-gzip-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(DEST/'compact-manifest.source.py').write_bytes(Path(__file__).read_bytes())
copy.unlink()
p=DEST/'README.md';s=p.read_text()
assert '(primary-run-manifest.json)' in s
s=s.replace('(primary-run-manifest.json)','(primary-run-manifest.json.gz)')
s=s.replace('All five raw JSONL files, complete source/test/input receipts and complete logs',
            'The complete run manifest is losslessly gzipped with a byte-verification receipt; the audited native run file is unchanged. All five raw JSONL files, complete source/test/input receipts and complete logs')
p.write_text(s)
p=ROOT/'configs/experiment_progress.json';d=json.loads(p.read_text())
d['readout_confirmation']['repository_manifest_lossless_gzip']=receipt
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
