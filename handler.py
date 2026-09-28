"""MuseTalk 1.5 pipeline worker; cached model + one character, serialized requests."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import wave

ROOT = Path('/opt/MuseTalk')
WORKER = 'musetalk-vertical-v1.0'
ENGINE = None
import threading
LOCK = threading.Lock()
COMMIT = '0a89dec45a0192b824e3cf4daf96c239440c5ed8'

def run(cmd, timeout=120, cwd=None, logfile=None):
    if logfile:
        with open(logfile, 'w') as out:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
                raise RuntimeError('MuseTalk timeout; see worker log')
        tail = Path(logfile).read_text(errors='replace')[-5000:]
        print(tail, flush=True)
        if code: raise RuntimeError('MuseTalk inference failed: '+tail)
        return tail
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError(r.stderr[-3000:])
    return r.stdout

def probe(path):
    return json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))

def decode(value, limit, label):
    if not isinstance(value,str) or len(value)>4*math.ceil(limit/3):
        raise ValueError(label+' missing or too large')
    try: data=base64.b64decode(value,validate=True)
    except Exception: raise ValueError(label+' invalid base64')
    if not data or len(data)>limit: raise ValueError(label+' invalid size')
    return data

def audio_seconds(data):
    with wave.open(io.BytesIO(data),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype()) != (16000,1,2,'NONE'):
            raise ValueError('Audio must be 16kHz mono PCM16 WAV')
        n=w.getnframes()
        if len(w.readframes(n))!=n*2: raise ValueError('Truncated WAV')
        duration=n/16000
    if not 0.08<=duration<=24.001: raise ValueError('Worker accepts audio 0.08-24s; pipeline partitions longer narration automatically')
    return duration

def handler(event):
    with LOCK:
        return process(event)

def process(event):
    global ENGINE
    req=event.get('input',{})
    if not isinstance(req,dict): raise ValueError('input must be an object')
    if req.get('action')=='diagnose':
        import torch
        manifest=ROOT/'model-manifest.json'
        return {'worker':WORKER,'commit':COMMIT,'cuda_available':torch.cuda.is_available(),
                'models_downloaded':manifest.is_file(), 'note':'Build check only; not GPU inference.'}
    if req.get('contract')!='vertical/1.0': raise ValueError('Wrong client/endpoint contract')
    clip=decode(req.get('video_base64'),4_000_000,'video')
    audio=decode(req.get('audio_base64'),1_000_000,'audio')
    duration=audio_seconds(audio)
    profile=req.get('gpu_profile','16gb');offset=req.get('frame_offset',0)
    if profile not in ('16gb','24gb'):raise ValueError('Invalid GPU profile')
    if type(offset) is not int or not 0<=offset<=1_000_000:raise ValueError('Invalid frame offset')
    config=json.dumps({'contract':'vertical/1.0','gpu_profile':profile,'frame_offset':offset},sort_keys=True,separators=(',',':'))
    input_hash=hashlib.sha256(clip+audio+config.encode()).hexdigest()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    started=time.monotonic()
    os.chdir(ROOT)
    model_hit=ENGINE is not None
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix='muse_') as folder:
        temp=Path(folder); source=temp/'seed.mp4'; wav=temp/'speech.wav'
        source.write_bytes(clip); wav.write_bytes(audio)
        info=probe(source); vs=[s for s in info['streams'] if s['codec_type']=='video']
        if len(vs)!=1: raise ValueError('Expected one video stream')
        v=vs[0]; num,den=map(int,v['r_frame_rate'].split('/'))
        seed_duration=float(info['format']['duration'])
        if (v['width'],v['height'])!=(720,1280) or den==0 or abs(num/den-25)>.001:
            raise ValueError('Client must normalize seed to 720x1280 / 25fps')
        if not 1<=seed_duration<=15.05: raise ValueError('Test seed must be 1-15s')
        if ENGINE is None:
            from engine import Engine
            ENGINE=Engine()
        generated=temp/'generated.mp4'
        metrics=ENGINE.generate(source,wav,generated,duration,profile,offset,hashlib.sha256(clip).hexdigest())
        gi=probe(generated); gv=next(s for s in gi['streams'] if s['codec_type']=='video')
        if (gv['width'],gv['height'])!=(720,1280): raise RuntimeError('Unexpected output framing')
        video_duration=float(gv.get('duration',gi['format']['duration']))
        if abs(video_duration-duration)>.12: raise RuntimeError('Incomplete inference video: '+str(video_duration))
        output=temp/'avatar.mp4'
        run(['ffmpeg','-nostdin','-y','-v','error','-i',str(generated),'-i',str(wav),
             '-map','0:v:0','-map','1:a:0','-vf','tpad=stop_mode=clone:stop_duration=0.2,fps=30',
             '-t',str(duration),'-c:v','libx264','-preset','fast','-crf','20','-maxrate','1250k',
             '-bufsize','2500k','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(output)])
        run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'])
        data=output.read_bytes()
        if len(data)>4_200_000: raise RuntimeError('Output exceeds inline response budget')
        metrics.update({'audio_seconds':duration,'seed_seconds':seed_duration,
                 'worker_seconds':round(time.monotonic()-started,3),'width':720,'height':1280,'output_fps':30,
                 'model_face_size':256,'source_fps':25,'loop':'forward_reverse','frame_offset':offset,
                 'gpu':torch.cuda.get_device_name(0),'gpu_total_gib':round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),
                 'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,2),
                 'peak_reserved_gib':round(torch.cuda.max_memory_reserved()/1024**3,2),
                 'precision':'fp16','model_cache_hit':model_hit,'gpu_profile':profile,
                 'upstream_commit':COMMIT,'usd_cost':None})
        print(json.dumps({'worker':WORKER,'metrics':metrics}),flush=True)
        return {'worker':WORKER,'input_sha256':input_hash,'sha256':hashlib.sha256(data).hexdigest(),
                'video_base64':base64.b64encode(data).decode(),'metrics':metrics}


if __name__=='__main__':
    import runpod
    runpod.serverless.start({'handler':handler})
