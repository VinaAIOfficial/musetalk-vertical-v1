import hashlib
import json
from pathlib import Path
import time
import urllib.request
import gdown
from huggingface_hub import hf_hub_download

ROOT=Path('/opt/MuseTalk')
REVISIONS=json.loads((Path(__file__).parent/'model_revisions.json').read_text())
SPECS={
 'TMElyralab/MuseTalk':('models',['musetalkV15/musetalk.json','musetalkV15/unet.pth']),
 'stabilityai/sd-vae-ft-mse':('models/sd-vae',['config.json','diffusion_pytorch_model.bin']),
 'openai/whisper-tiny':('models/whisper',['config.json','pytorch_model.bin','preprocessor_config.json']),
 'yzd-v/DWPose':('models/dwpose',['dw-ll_ucoco_384.pth']),
}
manifest={}
def record(p,source):
    if p.stat().st_size<100:raise RuntimeError('Download too small: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    manifest[str(p.relative_to(ROOT))]={'source':source,'sha256':h.hexdigest(),'bytes':p.stat().st_size}

def main():
    for repo,(folder,files) in SPECS.items():
        for name in files:
            p=Path(hf_hub_download(repo_id=repo,filename=name,revision=REVISIONS[repo],local_dir=str(ROOT/folder)))
            record(p,repo+'@'+REVISIONS[repo]); print('Downloaded',name,flush=True)
    extra={
      'models/face-parse-bisent/resnet18-5c106cde.pth':'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      'musetalk/utils/face_detection/detection/sfd/s3fd.pth':'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
      'models/face-parse-bisent/79999_iter.pth':'gdrive:154JgKpzCPW82qINcVieuPH3fZ2e0P812',
    }
    for name,url in extra.items():
        p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            try:
                if url.startswith('gdrive:'):
                    if not gdown.download(id=url.split(':',1)[1],output=str(p),quiet=False):raise RuntimeError('Face parsing download failed')
                else:
                    with urllib.request.urlopen(url,timeout=120) as src,p.open('wb') as out:
                        while chunk:=src.read(1024*1024):out.write(chunk)
                if p.stat().st_size<1_000_000:raise RuntimeError('Invalid checkpoint '+name)
                break
            except Exception:
                p.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2**attempt)
        record(p,url)
    (ROOT/'model-manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
