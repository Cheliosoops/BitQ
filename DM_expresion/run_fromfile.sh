start_time1=$(date +%s)
in1=$1            # ./run_fromfile.sh resnet18
input="./files_data/$in1.txt"
rm "./files_temporary/log.txt"
touch "./files_temporary/log.txt"
rm "./files_temporary/data.txt"
touch "./files_temporary/data.txt"
rm "./files_temporary/data.py"
touch "./files_temporary/data.py"
layer=0
Nthread=8
cnt=1
while IFS= read -r line 
do
    echo "$line"
    arr=($line)
    ((layer++))
    export OMP_NUM_THREADS=$Nthread
    python3 main.py ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]} ${arr[4]} ${arr[5]} ${arr[6]} 16 1 1 $Nthread 6 16 ${arr[7]} ${arr[8]} ${arr[9]} $layer  #> "$1.dirK1611/layer$cnt.cxy16-1-1_threads$OMP_NUM_THREADS.cnn$line.out"
    ((cnt++))
done < "$input"
end_time1=$(date +%s)
cost_time1=$[ $end_time1-$start_time1 ]
echo "run_fromfile_time: $(($cost_time1))s" >> ./files_temporary/data.txt

cp ./files_temporary/data.txt ./files_data/"$in1"_data_new.txt
cp ./files_temporary/data.py ./files_data/"$in1"_data_new.py
