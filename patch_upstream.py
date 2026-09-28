"""Fail loudly on missing faces, propagate errors, and check FFmpeg return codes."""
from pathlib import Path
p=Path('/opt/MuseTalk/scripts/inference.py')
s=p.read_text()
replacements={
    'if bbox == coord_placeholder:\n                    continue':
    "if bbox == coord_placeholder:\n                    raise RuntimeError('Face missing in a source frame; use a steady frontal seed')",
    'except:\n                    continue':
    "except Exception as exc:\n                    raise RuntimeError('Invalid face crop while blending') from exc",
    'print("Error occurred during processing:", e)': 'raise RuntimeError("MuseTalk task failed") from e',
    'os.system(cmd)': 'subprocess.run(cmd, shell=True, check=True)',
    'os.system(cmd_img2video)': 'subprocess.run(cmd_img2video, shell=True, check=True)',
    'os.system(cmd_combine_audio)': 'subprocess.run(cmd_combine_audio, shell=True, check=True)',
}
for old,new in replacements.items():
    if s.count(old)!=1: raise RuntimeError('Upstream changed: '+old)
    s=s.replace(old,new)
p.write_text(s)
# Above shell commands use only worker-generated fixed absolute paths, never user paths.
