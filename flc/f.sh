for g in 0 1 2  
do
for chi in 100 140 
do 
for dt in 0.001 0.005
do

echo "#!/usr/local_rwth/bin/zsh




#SBATCH --job-name=lcfermion$g.$chi.$dt   # The job name.
#SBATCH -c 4                    # The number of cpu cores to use.
#SBATCH --time=47:59:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=3200M        # The memory the job will use per cpu core.
#SBATCH --account=rwth0722
#SBATCH --mail-type=NONE
#SBATCH --mail-user=passetti@physik.rwth-aachen.de


#SBATCH --output=$HOME/output/flc.$g.$J.$V
#

### Change to the work directory
cd $HOME/Spinless_Boson/flc
### Execute your application
MKL_NUM_THREADS=4
export MKL_NUM_THREADS


export PYTHONPATH=$HOME/TeNPy
python3 time.py $g $chi $dt


echo \$?

date
" >$HOME/Spinless_Boson/Hubbard_GS/Bash/addBash_imp$g.$chi.$dt

sbatch <$HOME/Spinless_Boson/Hubbard_GS/Bash/addBash_imp$g.$chi.$dt

done
done
done
