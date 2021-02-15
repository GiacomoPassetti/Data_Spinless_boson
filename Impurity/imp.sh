for g in 0 0.5 1 2 
do
for omega in 10
do 
 

echo "#!/usr/local_rwth/bin/zsh




#SBATCH --job-name=lc_impurity__$g.$alpha   # The job name.
#SBATCH -c 4                    # The number of cpu cores to use.
#SBATCH --time=47:59:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=3200M        # The memory the job will use per cpu core.

#SBATCH --mail-type=NONE
#SBATCH --mail-user=passetti@physik.rwth-aachen.de


#SBATCH --output=$HOME/output/lc_imp_job.$g.$alpha
#

### Change to the work directory
cd $HOME/Spinless_Boson/Impurity
### Execute your application
MKL_NUM_THREADS=4
export MKL_NUM_THREADS


export PYTHONPATH=$HOME/TeNPy
python3 lc_impurity.py $g $omega 0


echo \$?

date
" >$HOME/Spinless_Boson/Impurity/Bash/addBash_imp$g.$omega

sbatch <$HOME/Spinless_Boson/Impurity/Bash/addBash_imp$g.$omega

done
done

