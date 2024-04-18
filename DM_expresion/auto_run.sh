echo "data check"
 
lie=0
lie1=0
hang1=0
error=0

in1=$1
echo "###  $in1  ###"
order=0
hang1=$(wc -l < ./files_data/$in1.txt)
echo $hang1
input="./files_data/${in1}_q.txt"
while IFS= read -r line
do
    touch "data_in_one_line.txt"  ###
    echo $line >> ./data_in_one_line.txt
    lie=$(wc -w data_in_one_line.txt | cut -d' ' -f1)
    rm "data_in_one_line.txt"     ###
    let lie1=$lie-1
    ((order++))
    if [ $hang1 -eq $lie1 ]; then
        echo "($hang1-$lie1)" $order "# right! #_"$order
    else
        echo "($hang1-$lie1)" $order "# error! ###################################################################################################################################_"$order
        ((error++))
    fi
done < "$input"


echo
echo "###  error_sum="$error"  ###"
echo
if [ $error -eq 0 ];then
    j=0
else
    sleep 60s
fi

rm "./files_temporary/run_out.txt"
touch "./files_temporary/run_out.txt"
rm "./files_temporary/run_out_all.txt"
touch "./files_temporary/run_out_all.txt"

echo "## $in1" >> ./files_temporary/run_out_all.txt
echo "######################################################################" >> ./files_temporary/run_out_all.txt

sleep 1s
echo "## run  " >> ./files_temporary/run_out.txt
. ./run_data.sh $in1 0.5 -0.5 ## net n r

echo


