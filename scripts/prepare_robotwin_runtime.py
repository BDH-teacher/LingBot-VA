"""Relocatable runtime overlay with a fixed simulation-step recorder hook."""
import difflib, json, os, pathlib, re, shutil, yaml
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT']);source=root/'third_party/RoboTwin-lingbot';target=root/'outputs/runtime/robotwin'
target.mkdir(parents=True,exist_ok=True)
for name in ('envs','task_config','description','script'):
    shutil.copytree(source/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
assets=source/'assets'
if not (assets/'embodiments/ur5-wsg/config.yml').exists():raise FileNotFoundError('UR5 assets missing; bash scripts/download_assets.sh')
# Generated embodiment config changes live only in the overlay.
asset_target=target/'assets';asset_target.mkdir(exist_ok=True)
for p in assets.iterdir():
    if p.name=='embodiments':continue
    dest=asset_target/p.name
    if not dest.exists():dest.symlink_to(p)
shutil.copytree(assets/'embodiments',asset_target/'embodiments',dirs_exist_ok=True)
for p in (asset_target/'embodiments').rglob('curobo*.yml'):
    if p.name.startswith('._'):continue
    data=yaml.safe_load(p.read_text().replace('${ASSETS_PATH}',str(target)));kin=data['robot_cfg']['kinematics']
    kin['asset_root_path']=str(p.parent)
    kin['urdf_path']=str(p.parent/kin['urdf_path'].split('/'+p.parent.name+'/')[-1]) if '/'+p.parent.name+'/' in kin['urdf_path'] else str(p.parent/kin['urdf_path'])
    kin['external_asset_path']=str(p.parent)
    kin['external_robot_configs_path']=str(p.parent)
    destination=p.with_name(p.name.replace('_tmp',''))
    destination.write_text(yaml.safe_dump(data,sort_keys=False))
p=target/'envs/_base_task.py';old=p.read_text()
new=re.sub(r'(?m)^([ \t]*)self\.scene\.step\(\)[ \t]*$',r'\1self.scene.step()\n\1if hasattr(self, "_after_sim_step"): self._after_sim_step()',old)
assert new!=old
compile(new,str(p),'exec')
p.write_text(new)
(root/'patches/robotwin_sim_clock.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/envs/_base_task.py',tofile='b/envs/_base_task.py')))
print('ROBOTWIN RUNTIME PREPARED',target)
