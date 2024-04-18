import sys
import subprocess
import math
from data_in import *
def get_score(acc, acc_max, acc_min, DM_DRAM, DM_SRAM, n, r, Energy_op, sum_operation, max_DM_sum, current_DM_sum):

    Edram=640
    Esram=5
    # Xacc=n*(acc-acc_min)/(acc_max-acc_min)+r
    # eXacc=math.exp(Xacc)
    sum_energy=(DM_DRAM*Edram+DM_SRAM*Esram)/10**12
    EE=sum_operation/sum_energy
    Jacc=math.log(acc)
    score = (100 - acc) / 100 +  0.2 * current_DM_sum / max_DM_sum 
    return score, sum_energy

def main():
    n=float(sys.argv[1])
    r=float(sys.argv[2])

    i=0
    ly_score={}
    print("\n")
    acc_list=[]
    acc_list1=[]

    DM_acc_list=[]
    score_list=[] 
    energy_list=[]  


    max_DM_sum = 0
    for key, value in list1.items():
        for one in value.values():
            DM_DRAM = one[3]
            DM_SRAM = one[0] + one[1] + one[2]
            current_DM_sum = DM_DRAM + DM_SRAM
            if current_DM_sum > max_DM_sum:
                max_DM_sum = current_DM_sum 

    for key,value in list1.items():
        acc_1=list(value)
        acc_list.append(acc_1[0])
        acc_list1.append(acc_1[0])

    acc_list1.sort()
    acc_max=acc_list1[-1]
    acc_min=acc_list1[0]
    print("acc_list=",acc_list)
    print("acc_max=",acc_max)
    print("acc_min=",acc_min)

    for key,value in list1.items():
        DM_acc_list.append(value)
    print("DM_acc_list=",DM_acc_list)
    for one in DM_acc_list:
        i=i+1
        acclist=list(one.keys())
        acc=acclist[0]
        DM_list=one[acc]
        DM_DRAM=DM_list[3]
        DM_SRAM=DM_list[0]+DM_list[1]+DM_list[2]
        current_DM_sum = DM_DRAM + DM_SRAM

        sum_operation=DM_list[3]
        Energy_op=DM_list[4]
        score, energy=get_score(acc, acc_max, acc_min, DM_DRAM, DM_SRAM, n, r, Energy_op, sum_operation, max_DM_sum, current_DM_sum)
        score_list.append(score)
        energy_list.append(energy)
       
        ly_score[i]=score

    min_score_index = score_list.index(min(score_list))
    min_score_energy = energy_list[min_score_index]
    print("The energy corresponding to the minimum score is:", min_score_energy)
    print("\nenergy_score=",energy_list)


    print("\nly_score=",ly_score)
    score_list.sort()
    for key,value in ly_score.items():
        if value==score_list[0]:
            bast_ly=key
    print("\nbast_ly=",bast_ly,"  bast_score=",score_list[-1]) 

    with open("./files_temporary/out_result.txt","at") as f:
        print("bast_score= ",score_list[-1],file=f)
    for key,value in list1.items():
        if key==bast_ly:
            print("\nbast_list=",key,":",value)

    subprocess.call('rm .//files_temporary//data_out.txt', shell=True)
    subprocess.call('touch .//files_temporary//data_out.txt', shell=True)
    f = open('.//files_temporary//data_in.txt','r+')
    line = f.readline()
    while line:
        if line[1]==' ':
            if int(line[0])==bast_ly:
                with open(".//files_temporary//data_out.txt","at") as ff:
                    print(line,end=" ",file=ff)

        elif line[2]==' ':
            k=int(line[0])*10+int(line[1])
            if k==bast_ly:
                # print(line)
                with open(".//files_temporary//data_out.txt","at") as ff:
                    print(line,end=" ",file=ff)

        line = f.readline()
    f.close()
    

if __name__ == "__main__":
    main()

 
