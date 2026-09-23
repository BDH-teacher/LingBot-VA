"""Create a runtime copy with an optional FlashAttention import; originals stay clean."""
import difflib, os, pathlib, shutil
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT'])
source=root/'third_party/lingbot-va'
target=root/'outputs/runtime/lingbot-va'
target.mkdir(parents=True,exist_ok=True)
for name in ('wan_va','evaluation'):
    shutil.copytree(source/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
p=target/'wan_va/modules/model.py'
old=p.read_text()
before='except:\n    from flash_attn import flash_attn_func'
after='''except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        def flash_attn_func(*args, **kwargs):
            raise RuntimeError("FlashAttention is not installed; use official torch or flex backend")'''
assert old.count(before)==1,'Upstream import changed; review required'
new=old.replace(before,after)
p.write_text(new)
(root/'patches/lingbot_optional_flash.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/wan_va/modules/model.py',tofile='b/wan_va/modules/model.py')))
print('RUNTIME PREPARED',target)
