for L in  20 40
do
for g0 in 0 1 2 3 4
do 
 

echo "#!/usr/local_rwth/bin/zsh




#SBATCH --job-name=gstark_$L.$g0  # The job name.
#SBATCH -c 4                    # The number of cpu cores to use.
#SBATCH --time=47:59:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=3200M        # The memory the job will use per cpu core.
#SBATCH --account=rwth0722
#SBATCH --mail-type=NONE
#SBATCH --mail-user=passetti@physik.rwth-aachen.de


#SBATCH --output=$HOME/output/ws.$L.$g0
#

### Change to the work directory
cd $HOME/Spinless_Boson/Quench_stark
### Execute your application
MKL_NUM_THREADS=4
export MKL_NUM_THREADS


export PYTHONPATH=$HOME/TeNPy
python3 ws.py $L $g0


echo \$?

date
" >$HOME/Spinless_Boson/Quench_stark/Bash/addBash_qs$L.$g0

sbatch <$HOME/Spinless_Boson/Quench_stark/Bash/addBash_qs$L.$g0

done
done

