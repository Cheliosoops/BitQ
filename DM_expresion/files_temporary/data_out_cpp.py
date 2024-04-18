import sys
sys.path.append(r"/home/wpliang/lwp/ASPLOS_artifact-master/TileLoopGenerator")
from files_data.googlenet_data import *
from files_temporary.run_function import *
import subprocess
def main():
    gen_config=gen_config57_8
    select_fork=select_fork57_8
    qw=8
    Nb=1
    Nf=128
    Nx=7
    Ny=7
    Nc=832
    Nw=1
    Nh=1
    kf=16
    ic=1
    of=1
    numA=6
    numB=16
    strideX=1
    strideY=1
    print(gen_config, select_fork)
    runner = Runner(pbsz=[Nb, Nf,Nx,Ny,Nc,Nw,Nh],split_fc=[kf,ic,of],config=gen_config, parallel_fork=select_fork, numAB=[numA,numB], strideXY=[strideX,strideY], qw=qw)
    runner.run_one_from_batch()
if __name__ == "__main__":
    main()
