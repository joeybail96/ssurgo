#!/bin/bash

#SBATCH --job-name=plot_playa_mask
#SBATCH --output=plot_playa_mask.log
#SBATCH --mail-user=joey.bail@utah.edu
#SBATCH --mail-type=ALL

#SBATCH --account=haskins
#SBATCH --partition=notchpeak-shared

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=0-06:00       
#SBATCH --mem=60G            

###############################################################################

echo My working directory is `pwd`
echo -e '\t'`hostname` at `date`
echo

start_date=$(date)

module use ~/MyModules; module load micromamba/latest;

micromamba activate gcpy

python /uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/plot.py

end_date=$(date)

echo
echo "Start: "$start_date
echo "End: "$end_date



