in1=$1  #resnet18

order2=0
layer2=1
lie3=0
lie2=0
error2=0
let hang2=$(wc -l ./files_data/$in1.txt)
input=./files_data/"$in1"_q.txt
echo "###  input and qw text  ###"
while IFS= read -r line
do
    rm "./files_temporary/data_in_one_line.txt"
    touch "./files_temporary/data_in_one_line.txt"
    echo $line >>./files_temporary/data_in_one_line.txt
    let lie3=$(wc -w ./files_temporary/data_in_one_line.txt)
    let lie2=$lie3-1
    echo "$line2" "($hang2-$lie2)"
    ((order2++))
    if [ $hang2 -eq $lie2 ]; then
        echo $order2 $line "# right! #_"$order2
    else
        echo $order2 $line "# error! ###################################################################_"$order2
        ((error2++))
    fi
done < "$input"
echo
echo "###  error_sum="$error2"  ###"
echo
if [ $error2 -eq 0 ];then
    j=0
else
    sleep 10s
    echo "#######################"
fi



rm "./files_temporary/data_in.txt"
touch "./files_temporary/data_in.txt"
rm "./files_temporary/data_in.py"
touch "./files_temporary/data_in.py"
echo -n "list1={" >> ./files_temporary/data_in.py
order=0
acc=0
hang=0
lie=0
sum_op=0
E_op=0
arr_op=(0 0 0 0 0 0 0 0 23 0 0 0 0 0 0 0 95 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 400)  # *100
arr_bw=(65 57 28 16)
let lie=$lie2-1
let hang=$(wc -l ./files_data/$in1_q.txt)
echo "num="$hang "ly="$lie
input=./files_data/"$in1"_q.txt
while IFS= read -r line
do
    DM0=0
    DM1=0
    DM2=0
    DM4=0
    arr1=($line) #acc,l1,l2,l3,.....
    ((order++))
    echo $line
    layer_i=0
    for ((i=lie;i>0;i-=1))
    do
        ((layer_i++))
        input=./files_data/"$in1"_data.txt
        while IFS= read -r line
        do
            op=0
            arr2=($line) #layer,qw,DV,lv,..... 8  16  463423  74106 74106 60244 4 1 0 2 
            ##echo $line
            if [ $layer_i -eq ${arr2[0]} ] && [ ${arr1[layer_i]} -eq ${arr2[1]} ]; then
                echo $order ${arr2[0]} ${arr2[1]} ${arr1[0]} ${arr2[2]} ${arr2[3]} ${arr2[4]} ${arr2[5]} ${arr2[6]} ${arr2[7]} ${arr2[8]}  ${arr2[9]}  #${arr2[*]:3}
                echo $order ${arr2[0]} ${arr2[1]} ${arr1[0]} ${arr2[*]:2} >> ./files_temporary/data_in.txt
                
                if [ ${arr2[6]} -eq 0 ]; then
                    let DM0=DM0+${arr2[2]%.*}*${arr_bw[0]}
                elif [ ${arr2[6]} -eq 1 ]; then
                    let DM1=DM1+${arr2[2]%.*}*${arr_bw[1]}
                elif [ ${arr2[6]} -eq 2 ]; then
                    let DM2=DM2+${arr2[2]%.*}*${arr_bw[2]}
                elif [ ${arr2[6]} -eq 4 ]; then
                    let DM4=DM4+${arr2[2]%.*}*${arr_bw[3]}
                fi
                if [ ${arr2[7]} -eq 0 ]; then
                    let DM0=DM0+${arr2[3]%.*}*${arr_bw[0]}
                elif [ ${arr2[7]} -eq 1 ]; then
                    let DM1=DM1+${arr2[3]%.*}*${arr_bw[1]}
                elif [ ${arr2[7]} -eq 2 ]; then
                    let DM2=DM2+${arr2[3]%.*}*${arr_bw[2]}
                elif [ ${arr2[7]} -eq 4 ]; then
                    let DM4=DM4+${arr2[3]%.*}*${arr_bw[3]}
                fi
                if [ ${arr2[8]} -eq 0 ]; then
                    let DM0=DM0+${arr2[4]%.*}*${arr_bw[0]}
                elif [ ${arr2[8]} -eq 1 ]; then
                    let DM1=DM1+${arr2[4]%.*}*${arr_bw[1]}
                elif [ ${arr2[8]} -eq 2 ]; then
                    let DM2=DM2+${arr2[4]%.*}*${arr_bw[2]}
                elif [ ${arr2[8]} -eq 4 ]; then
                    let DM4=DM4+${arr2[4]%.*}*${arr_bw[3]}
                fi
                if [ ${arr2[9]} -eq 0 ]; then
                    let DM0=DM0+${arr2[5]%.*}*${arr_bw[0]}
                elif [ ${arr2[9]} -eq 1 ]; then
                    let DM1=DM1+${arr2[5]%.*}*${arr_bw[1]}
                elif [ ${arr2[9]} -eq 2 ]; then
                    let DM2=DM2+${arr2[5]%.*}*${arr_bw[2]}
                elif [ ${arr2[9]} -eq 4 ]; then
                    let DM4=DM4+${arr2[5]%.*}*${arr_bw[3]}
                fi

                let op=${arr2[10]}*${arr2[11]}*${arr2[12]}*${arr2[13]}*${arr2[14]}*${arr2[15]}*${arr2[16]}
                let sum_op=sum_op+op
                acc=${arr1[0]}
                qq=${arr1[$layer_i]}
                let E_op=E_op+op*${arr_op[$qq]}/100
                if [ $layer_i -eq $lie ];then
                    echo "############  num=$order DM0= $DM0 DM1= $DM1 DM2= $DM2 DM4= $DM4 acc= $acc sum_op=$sum_op E_op=$E_op"
                    #echo -n "'"$order"':{\""$DV_sum"\","$acc"}," >> ./files_temporary/data_in.py
                    echo -n "$order:{$acc:[$DM0,$DM1,$DM2,$DM4,$sum_op,$E_op]}," >> ./files_temporary/data_in.py
                    let DV_sum=0
                fi

            fi

        done < "$input"
    done
done < "$input"
sed -i '$s/.$//' ./files_temporary/data_in.py
echo -n "}" >> ./files_temporary/data_in.py



rm "./files_temporary/out_result.txt"
touch "./files_temporary/out_result.txt"
printf "\n### $in1 ###\n" >> ./files_temporary/out_result.txt
n=$2
r=$3
python3 ./files_temporary/data_out.py $n $r ##########################################################################################
sleep 2s



