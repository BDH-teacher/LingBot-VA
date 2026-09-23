"""Run an argument vector without shell interpolation; append audited exit status."""
import datetime, json, os, pathlib, shlex, subprocess, sys, time
root = pathlib.Path(os.environ.get('LINGBOT_UR5_ROOT', pathlib.Path(__file__).resolve().parents[1]))
start = time.time()
record = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'cwd': os.getcwd(), 'argv': sys.argv[1:], 'command': shlex.join(sys.argv[1:])}
print('+ ' + record['command'], flush=True)
try:
    result = subprocess.run(sys.argv[1:])
    code = result.returncode
except (OSError, KeyboardInterrupt) as exc:
    print(str(exc), file=sys.stderr)
    code = 130 if isinstance(exc, KeyboardInterrupt) else 127
record.update(exit_code=code, elapsed_seconds=round(time.time()-start, 3))
with (root/'logs/commands.jsonl').open('a') as f:
    f.write(json.dumps(record, ensure_ascii=False)+'\n')
sys.exit(code)
