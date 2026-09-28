"""Persistent MuseTalk 1.5 engine, using the pinned upstream normal-inference blend.
Only the most recently prepared character is cached; all full frames/latents stay on CPU.
"""
import gc
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import time

class Engine:
    def __init__(self):
        import torch
        from transformers import WhisperModel
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        self.torch=torch;self.device=torch.device('cuda:0');self.dtype=torch.float16
        self.vae,self.unet,self.pe=load_all_model(device=self.device)
        self.vae.vae.to(self.device,dtype=self.dtype).eval()
        self.unet.model.to(self.device,dtype=self.dtype).eval()
        self.pe.to(self.device,dtype=self.dtype).eval()
        self.whisper=WhisperModel.from_pretrained('models/whisper').to(self.device,dtype=self.dtype).eval().requires_grad_(False)
        self.processor=AudioProcessor(feature_extractor_path='models/whisper')
        self.fp=FaceParsing(left_cheek_width=90,right_cheek_width=90)
        self.timesteps=torch.tensor([0],device=self.device)
        self.character=None;self.frames=[];self.boxes=[];self.latents=[]

    def prepare(self,source,character):
        if self.character==character:return True
        import cv2
        from musetalk.utils.preprocessing import get_landmark_and_bbox,coord_placeholder
        self.character=None;self.frames=[];self.boxes=[];self.latents=[]
        gc.collect();self.torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='frames_') as folder:
            cap=cv2.VideoCapture(str(source));paths=[]
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    if len(paths)>=376:raise ValueError('Too many seed frames')
                    p=Path(folder)/f'{len(paths):08}.png'
                    if not cv2.imwrite(str(p),frame):raise RuntimeError('Cannot write seed frame')
                    paths.append(str(p))
            finally:cap.release()
            if not paths:raise ValueError('Empty seed video')
            boxes,frames=get_landmark_and_bbox(paths,0)
            if len(boxes)!=len(frames):raise RuntimeError('Face detector frame mismatch')
            latents=[];clean_boxes=[]
            with self.torch.inference_mode():
                # Stable preparation on worker restart; VAE upstream samples its distribution.
                self.torch.manual_seed(int(character[:8],16))
                for i,(box,frame) in enumerate(zip(boxes,frames)):
                    if tuple(box)==coord_placeholder:raise ValueError(f'No face in seed frame {i}; choose a clearly visible face throughout the clip')
                    x1,y1,x2,y2=map(int,box);y2=min(y2+10,frame.shape[0])
                    if not (0<=x1<x2<=frame.shape[1] and 0<=y1<y2<=frame.shape[0]):raise ValueError(f'Invalid face crop in frame {i}')
                    face=cv2.resize(frame[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4)
                    latents.append(self.vae.get_latents_for_unet(face).detach().cpu())
                    clean_boxes.append((x1,y1,x2,y2))
            self.frames=frames;self.boxes=clean_boxes;self.latents=latents;self.character=character
        return False

    def frame_index(self,index):
        n=len(self.frames);i=index%(2*n)
        return i if i<n else 2*n-1-i

    def render(self,wav,output,seconds,batch,offset):
        import cv2
        import numpy as np
        from musetalk.utils.blending import get_image
        torch=self.torch
        with torch.inference_mode():
            features,length=self.processor.get_audio_feature(str(wav),weight_dtype=self.dtype)
            prompts=self.processor.get_whisper_chunk(features,self.device,self.dtype,self.whisper,length,fps=25)
            expected=math.floor(seconds*25)
            if len(prompts)!=expected or expected<1:raise RuntimeError('Audio feature/frame count mismatch')
            prompts=prompts.cpu()
            with tempfile.TemporaryFile(mode='w+b') as errors:
                proc=subprocess.Popen(['ffmpeg','-nostdin','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24',
                    '-s','720x1280','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
                    '-crf','18','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE,stderr=errors)
                try:
                    for start in range(0,len(prompts),batch):
                        end=min(start+batch,len(prompts))
                        indices=[self.frame_index(offset+i) for i in range(start,end)]
                        latent=torch.cat([self.latents[i] for i in indices],dim=0).to(self.device,dtype=self.dtype)
                        audio=self.pe(prompts[start:end].to(self.device,dtype=self.dtype))
                        pred=self.unet.model(latent,self.timesteps,encoder_hidden_states=audio).sample
                        images=self.vae.decode_latents(pred)
                        del latent,audio,pred
                        for idx,face in zip(indices,images):
                            box=self.boxes[idx];x1,y1,x2,y2=box
                            face=cv2.resize(face.astype(np.uint8),(x2-x1,y2-y1))
                            frame=get_image(self.frames[idx].copy(),face,box,mode='jaw',fp=self.fp)
                            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                        if start%100==0:print(f'MuseTalk frames {end}/{len(prompts)} batch={batch}',flush=True)
                    proc.stdin.close()
                    if proc.wait(timeout=120):
                        errors.seek(0);raise RuntimeError('Encoder failed: '+errors.read().decode(errors='replace')[-2000:])
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
                    if proc.stdin and not proc.stdin.closed:
                        try:proc.stdin.close()
                        except OSError:pass  # Preserve the inference/OOM exception during encoder cleanup.

    def generate(self,source,wav,output,seconds,profile,offset,character):
        started=time.monotonic();hit=self.prepare(source,character)
        prep=time.monotonic()-started;batch=2 if profile=='16gb' else 4;retries=0
        while True:
            failed=False
            try:
                self.render(wav,output,seconds,batch,offset)
            except self.torch.cuda.OutOfMemoryError:
                if batch==1:raise RuntimeError('GPU out of memory even at batch 1. Retry with more VRAM; no new job was submitted automatically.') from None
                failed=True
            if not failed:break
            # Outside the exception handler so the traceback and GPU tensors have been released.
            batch=max(1,batch//2);retries+=1;gc.collect();self.torch.cuda.empty_cache()
            print(f'GPU OOM: retrying within SAME request, batch={batch}',flush=True)
        return {'character_cache_hit':hit,'preparation_seconds':round(prep,3),'batch_size':batch,'oom_retries':retries,
                'inference_seconds':round(time.monotonic()-started-prep,3)}
