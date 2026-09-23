import argparse
from lingbot_ur5.paths import ROOT
from lingbot_ur5.dataset import convert
p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',default=str(ROOT/'datasets/ur5_move_can_pot'))
p.add_argument('--instruction',default='Pick up the can with the left arm and place it beside the pot.')
a=p.parse_args();convert(a.raw,a.output,a.instruction)
