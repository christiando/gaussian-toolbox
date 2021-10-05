input="new_hsk.txt"
while IFS= read -r line
do
  $line
done < "$input"

