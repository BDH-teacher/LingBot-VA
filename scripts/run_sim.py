import argparse,datetime,pathlib,tempfile
from lingbot_ur5.paths import ROOT
from lingbot_ur5.sim import collect

def create_sim_test_output():
    parent=ROOT/'outputs/ur5_sim_test';parent.mkdir(parents=True,exist_ok=True)
    prefix=datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_')
    return pathlib.Path(tempfile.mkdtemp(prefix=prefix,dir=parent))

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=1);p.add_argument('--task',choices=['move_can_pot'],default='move_can_pot')
    p.add_argument('--output-dir',type=str,help='Explicit destination; default creates outputs/ur5_sim_test/run_<time>_<id>')
    p.add_argument('--headless',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--randomization',action='store_true')
    a=p.parse_args()
    if a.episodes<1: p.error('episodes must be positive')
    output=pathlib.Path(a.output_dir).expanduser().resolve() if a.output_dir else create_sim_test_output()
    print('SIM OUTPUT DIRECTORY:',output,flush=True)
    collect(a.episodes,output,not a.headless,a.seed,a.randomization)
if __name__=='__main__':main()
