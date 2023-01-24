#!/bin/bash -l
#SBATCH --job-name=task2vec_vision
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:20:00
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test_serial-job.%A_%a.out
#SBATCH --error=test_serial-job.%A_%a.error

while getopts e:n:s: flag
do
    case "${flag}" in
        e) exp_name=${OPTARG};;
        n) exp_notes=${OPTARG};;
        s) skip_layer=${OPTARG};;
    esac
done

echo 'Activate Environment'
source ~/task2vec/bin/activate

cd $(ws_find zap_hpo_og)/task2vec/

python3 -m zap_meta --exp_name $exp_name --exp_notes $exp_notes --skip_layer $skip_layer


source deactivate