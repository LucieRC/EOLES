##############
### ****** ###
##############

### A file for shared variables between all scripts ###

## Name of the model for output files ##
model_name = "test"

## Choice of input years ##
# You can currently add any year(s) between 2000 and 2018. Order will be kept.
# Data are extracted from renewable.ninja and stored in the vre_profiles.csv file.
# Available data are 2000-2019 for the V90, V100, pv_EW and pv_S technologies, 2000-2018 for the others.
# Recommanded values : 2006 (1y), 2005/2006 (2y), 2004/2005/2006 (3y). For all 19 years, use range(2000,2019).
# The first nb_years values of this list will be used.
input_years=[2018]

## Number of years ##
# Be aware that NPP outage planning will behave differently in each scenario.
# NPP fuel cycle length will be set to 2 years if nb_years is even, 1.5 years if nb_years is a multiple of 3.
# WARNING : for that reason, the nuc_opti module will only run with multiples of 2 or 3.
# You can set this value to len(input_years), or overide it to use the first nb_years values of the input_years.
nb_years = len(input_years)

demand_profile = "2050_RTE_580TWh"

##############
### ****** ###
##############