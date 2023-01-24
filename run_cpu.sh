#!/bin/bash -l

while getopts e:n:s: flag
do
    case "${flag}" in
        e) exp=${OPTARG};;
        n) exp_notes=${OPTARG};;
        s) skip_layer=${OPTARG};;
    esac
done

echo "Activate Env"

for i in 1 2 3 4 5 6 7 8 9 10 11
do
  exp_name="${exp}_${i}"
  echo " Run Params $exp_notes $exp_name Layer $i"
  python3 -m zap_meta --exp_name $exp_name --exp_notes $exp_notes --skip_layer $i
done

echo "Done"