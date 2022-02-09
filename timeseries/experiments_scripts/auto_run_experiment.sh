input="run_exp1_0.75tr_05102021.txt"
while IFS= read -r line
do
  $line
done < "$input"

