for g in 0 
do
for J in 1
do 
for V in 2
do

echo "#!/usr/local_rwth/bin/zsh




#SBATCH --job-name=gs_hub$g.$J.$V   # The job name.
#SBATCH -c 4                    # The number of cpu cores to use.
#SBATCH --time=47:59:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=3200M        # The memory the job will use per cpu core.

#SBATCH --mail-type=NONE
#SBATCH --mail-user=passetti@physik.rwth-aachen.de


#SBATCH --output=$HOME/output/Hubbard.$g.$J.$V
#

### Change to the work directory
cd $HOME/Spinless_Boson/Hubbard_GS
### Execute your application
MKL_NUM_THREADS=4
export MKL_NUM_THREADS


export PYTHONPATH=$HOME/TeNPy
python3 hubbard_GS.py $g $J $V


echo \$?

date
" >$HOME/Spinless_Boson/Hubbard_GS/Bash/addBash_imp$g.$J.$V

sbatch <$HOME/Spinless_Boson/Hubbard_GS/Bash/addBash_imp$g.$J.$V

done
done
done

