"""Build CuRobo at a compatible release; record all local dependency changes."""
import difflib, json, os, pathlib, subprocess, sys
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT'])
repo=root/'third_party/curobo'
if not repo.exists(): subprocess.run(['git','clone','https://github.com/NVlabs/curobo.git',str(repo)],check=True)
lock_path=root/'configs/sources.lock.json'
lock=json.loads(lock_path.read_text())
ref=lock.get('repos',{}).get('curobo',{}).get('commit','v0.7.6')
subprocess.run(['git','-C',str(repo),'checkout',ref],check=True)
lock['repos']['curobo']={'url':'https://github.com/NVlabs/curobo.git','commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'tag':'v0.7.6'}
lock_path.write_text(json.dumps(lock,indent=2)+'\n')
env=os.environ.copy()
import torch
arch=os.environ.get('TORCH_CUDA_ARCH_LIST')
if not arch:
    if not torch.cuda.is_available():raise RuntimeError('Set TORCH_CUDA_ARCH_LIST when building without a visible GPU')
    major,minor=torch.cuda.get_device_capability();arch=f'{major}.{minor}'
env.update(CUDA_HOME=sys.prefix,MAX_JOBS=os.environ.get('MAX_JOBS','2'),TORCH_CUDA_ARCH_LIST=arch)
print('CUROBO BUILD CUDA ARCH',arch,flush=True)
env['CC']=str(pathlib.Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-cc')
env['CXX']=str(pathlib.Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-c++')
subprocess.run([sys.executable,'-m','pip','install','setuptools_scm==10.2.3','pybind11==3.1.0'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-e',str(repo),'--no-build-isolation','-c',str(root/'envs/robotwin-requirements.txt')],env=env,check=True)
# Apply only the official UTF-8 URDF file-reading fix. Preserve collision checking.
import sapien
p=pathlib.Path(sapien.__file__).parent/'wrapper/urdf_loader.py'
old=p.read_text();new=old.replace('open(urdf_file, "r")','open(urdf_file, "r", encoding="utf-8")').replace('open(srdf_file, "r")','open(srdf_file, "r", encoding="utf-8")')
if old!=new:
    p.write_text(new)
    (root/'patches/sapien_utf8.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/sapien/wrapper/urdf_loader.py',tofile='b/sapien/wrapper/urdf_loader.py')))
# Editable .pth files are processed on interpreter startup, not in this installer.
subprocess.run([sys.executable,'-c','from curobo.wrap.reacher.motion_gen import MotionGen; import curobo; print("CUROBO IMPORT OK", curobo.__file__)'],env=env,check=True)
