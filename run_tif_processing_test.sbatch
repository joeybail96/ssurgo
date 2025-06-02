#!/bin/bash

#SBATCH --job-name=tif_regrid_template_test
#SBATCH --output=tif_regrid_template_test.log
#SBATCH --mail-user=joey.bail@utah.edu
#SBATCH --mail-type=ALL

#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=0-00:05
#SBATCH --mem=1G        

###############################################################################

echo My working directory is `pwd`
echo -e '\t'`hostname` at `date`
echo

start_date=$(date)

module use ~/MyModules; module load micromamba/latest;

micromamba activate gcpy

python /uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/sbatch_test.py

end_date=$(date)

echo
echo "Start: "$start_date
echo "End: "$end_date



