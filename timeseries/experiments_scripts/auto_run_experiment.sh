input="commands_to_run.txt"
while IFS= read -r line
do
  $line
done < "$input"

