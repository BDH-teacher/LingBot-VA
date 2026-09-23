import pathlib,os
import numpy as np
import sapien
from PIL import Image
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT'])
engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
scene=engine.create_scene();scene.set_ambient_light([0.5,0.5,0.5]);scene.add_directional_light([1,0,-1],[1,1,1])
scene.add_ground(0)
material=renderer.create_material();material.set_base_color([0.8,0.1,0.1,1.0])
b=scene.create_actor_builder();b.add_box_visual(half_size=[0.1,0.1,0.1],material=material);actor=b.build_static();actor.set_pose(sapien.Pose([1,0,0.1]))
cam=scene.add_camera('test',320,240,1.0,0.1,10);cam.set_pose(sapien.Pose([0,0,0.3]))
scene.update_render();cam.take_picture();rgb=(cam.get_picture('Color')[:,:,:3]*255).clip(0,255).astype('uint8')
assert rgb.std()>1,'Renderer returned blank frame'
p=root/'outputs/renderer.png';p.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(p)
print('SAPIEN RENDER OK',p)
