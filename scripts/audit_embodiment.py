import hashlib,json,os,pathlib,xml.etree.ElementTree as ET,zipfile
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT']);prefix='embodiments/ur5-wsg/'
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import JOINT_NAMES
with zipfile.ZipFile(root/'third_party/RoboTwin-lingbot/assets/embodiments.zip') as archive:
    raw=archive.read(prefix+'ur5_wsg_gripper.urdf');urdf=ET.fromstring(raw)
    joints={j.attrib['name']:j for j in urdf.findall('joint')}
    limits={k:[] for k in ('lower','upper','velocity','effort')}
    for name in JOINT_NAMES:
        limit=joints[name].find('limit')
        for key in limits:limits[key].append(float(limit.attrib[key]))
    meshes=[e.attrib['filename'] for e in urdf.findall('.//mesh')]
    for mesh in meshes:
        member=prefix+mesh.removeprefix('./')
        if member not in archive.namelist():raise FileNotFoundError(member)
    data={'asset':'ur5-wsg','joint_names':list(JOINT_NAMES),'arm_dof':6,'joint_limits':limits,
          'urdf_sha256':hashlib.sha256(raw).hexdigest(),'meshes':meshes,'meshes_verified':True,
          'srdf_exists':prefix+'ur5.srdf' in archive.namelist(),'eef_joint':'ee_fixed_joint','move_group':'ee_link',
          'gripper':'WSG base_joint_gripper_left/right','source_zip_revision':json.loads((root/'configs/sources.lock.json').read_text())['huggingface']['TianxingChen/RoboTwin2.0']['revision']}
(root/'configs/sim/ur5_asset_audit.json').write_text(json.dumps(data,indent=2));print('UR5 ASSET AUDIT OK',len(meshes),'mesh references')
