# -*- coding: utf-8 -*-

import os
import pandas as pd
from CONFIG import model_name

###############################
##### RUN THE EOLES MODEL #####
###############################

# This file should be used to run the EOLES model including several corrections for nuclear flexibilty
# For standard use, you may change all inputs files and the CONFIG.py file
# The nuclear optimisation module also has it own parameters in its file
# Shared variables are defined in the CONFIG file

################
##### CODE #####
################

# Delete any previous temporary inputs that were not cleared if the programm did not exited correctly
while(True):
    in_fix_capa = pd.read_csv("inputs/fix_capa.csv", index_col=0, header=None).squeeze("columns")
    try:
        in_fix_capa.drop("nuc", inplace=True)
        in_fix_capa.to_csv(path_or_buf='./inputs/fix_capa.csv', header= False)
    except Exception as e:
        break

# Run preprocess (simplified EOLES) to get to optimal nuclear capacity & production deficit
print("\n[--*-- PREPROCESS --*--]\n")
print("Running preprocess...")
exec(open('Eoles_elec_vf_preprocess.py').read())

# Run nuclear optimisation module
print("\n[--*-- NUC OPTI MODULE --*--]\n")
exec(open('Eoles_nuc_opti.py').read())

# Run EOLES
print("\n[--*-- EOLES --*--]\n")
print("Running the EOLES model...")
exec(open('Eoles_elec_vf.py').read())

# Delete temporary files & inputs
if os.path.isfile("./outputs/eoles_"+model_name+"_hourly_deficit_prod.csv"):
    os.remove("./outputs/eoles_"+model_name+"_hourly_deficit_prod.csv")
else:
    print("Error: ./outputs/eoles_"+model_name+"_hourly_deficit_prod.csv file not found")

in_fix_capa = pd.read_csv("inputs/fix_capa.csv", index_col=0, header=None).squeeze("columns")
in_fix_capa.drop("nuc", inplace=True)
in_fix_capa.to_csv(path_or_buf='./inputs/fix_capa.csv', header= False)