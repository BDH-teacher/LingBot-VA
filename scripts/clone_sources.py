import json,os,pathlib,subprocess
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT']);repos=json.loads((root/'configs/sources.lock.json').read_text())['repos']
for name,spec in repos.items():
    p=root/'third_party'/name
    if p.exists():
        sha=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if sha!=spec['commit']:raise RuntimeError(f'{name} checkout differs from lock; review locally, expected {spec["commit"]}')
        continue
    if name=='RoboTwin-lingbot':
        subprocess.run(['git','-C',str(root/'third_party/RoboTwin'),'worktree','add',str(p),spec['commit']],check=True)
    else:
        subprocess.run(['git','clone',spec['url'],str(p)],check=True)
        subprocess.run(['git','-C',str(p),'checkout',spec['commit']],check=True)
print('PINNED SOURCES VERIFIED')
