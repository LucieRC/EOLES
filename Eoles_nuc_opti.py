# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from CONFIG import model_name, nb_years

"""
Nuclear power plant outage for refuelling and maintenance scheduling module for the EOLES family of models
Written by Quentin Bustarret (July 2022)
"""

##########################
### OUTAGES SCHEDULING ###
##########################

# First implementation : Brute search
# Result : Inefficient

# Second implementation : calculating the minimal use rate 
# Result : Unexpected behaviour

# Third implementation : calculating the production deficit
# Result : Current version, works great

# Forth implementation : using a LP model
# Result : Not linear, abandoned

### INPUTS ###

ini_deficit_prod = pd.read_csv("./outputs/eoles_" + model_name + "_hourly_deficit_prod.csv",usecols=["hour","deficit"])
capa = pd.read_csv("./outputs/eoles_" + model_name + "_hourly_deficit_prod.csv",usecols=["hour","nuc_capa"])
capa = float(capa.at[0,'nuc_capa']) # in GW, result from EOLES
#capa -= 1.6 # We consider that resticting nuc flexibility implies EOLES installing 1 less EPR to obtain the economic optimum. In theory, EOLES should be run with nuc_capa + x*1.6 xith x in [-3,-2,-1,0,1,2,3] to see which results in the optimal cost, but involves way too much computation. Empiricaly, the value -1 is the closest to the economic optimum. 

capa_per_plant = 1.6 # in GW. 1,6=EPRs

minimal_operating_power = 0.25 # % of Pnom, minimal operating capacity. 0.2=current fleet , 0.25=EPRs. Value found in EPR documentation
power_outage_start = 0.8 # % of Pnom, active capacity when outage starts. Estimated from A. Lynch et al. (2022).

len_outage = 24*30 # in hours. This is an estimate for EPRs that takes into account standard refuelling and bigger maintenances
len_fixed_output = 24*30  # in hours. Estimated from A. Lynch et al. (2022)
len_limited_flexibility = 2*24*30  # in hours. Estimated from A. Lynch et al. (2022)

# In hours. The module will separate consecutives outages by an average value +- delta. 
# EPR documentation indicates 2*delta = 70 EFPD.
# 0 is a valid value for the delta parameter if you wish not to use it
delta = 24 

#Use plots only to test this module, as it will wait for you to close the window before exporting outputs
With_plots = False # Allow all plots and display the final plot
Step_by_step_plots = False # A plot at each iteration (for each NPP)

#######
# According to A. Lynch et al. (2022), the current minimum available nuclear fleet is fixed at 50% and no more than 10% of the fleet shall go off the same week.
# These constraint are implemented here but are subject to debate.
# If you wish NOT to take them into account, set the following value to False.

With_Lynch_constraints = True

max_share_weekly_nuc_outage = 0.1 # Value is found in the article
min_share_online_npp = 0.5 # Value is found in the article
impossible_values = pd.DataFrame(data={'hour': [i for i in range(8760*nb_years)], 'value': [max(int(max_share_weekly_nuc_outage*int(np.floor(capa/capa_per_plant))),1) for i in range(8760*nb_years)]})
#######

### CODE ###

start = time.time()
print("Installed NPP capacity (GW): "+str(round(int(np.floor(capa/capa_per_plant))*capa_per_plant,4)))
print("Scheduling NPP shutdowns for refuelling and maintenance...")

if nb_years == 1:
    nb_outages = 1 # Number of outages per NPP
    step = 8760 # Time between outages. Current fleet : 1 year, EPRs up to 2 years in theory but most likely 36 months
elif nb_years % 2 == 0:
    nb_outages = int(nb_years/2)
    step = 8760 * 2
elif nb_years % 3 == 0: 
    nb_outages = int(nb_years/1.5)
    step = int(8760 * 1.5)
else:
    raise ValueError("The number of years must either be 1 or a multiple of 2 or 3. Please change the value in the CONFIG.py file")

months=["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
date=[]
year=2050
for month in range(12*nb_years):
    current_month = month if month < 12 else month - 12*(year-2050)
    for day in range(31):
        if (day > 27 and current_month == 1): continue
        if (day == 30 and (current_month == 3 or current_month == 5 or current_month == 8 or current_month == 10)): continue
        for hour in range(24):
            date.append(str(day+1)+" "+months[current_month]+" "+str(year)+" "+str(hour)+"h")
        if current_month == 11 and day == 30:
            year += 1

def aggregate(deficit_prod):
    aggregated_deficit_prod=deficit_prod.copy()
    aggregated_deficit_prod.at[0,'deficit']=sum(deficit_prod.at[j,'deficit'] for j in range(len_outage))
    for i in range(1,deficit_prod.shape[0]):
        if i <= deficit_prod.shape[0]-len_outage:
            aggregated_deficit_prod.at[i,'deficit']=aggregated_deficit_prod.at[i-1,'deficit']-deficit_prod.at[i-1,'deficit']+deficit_prod.at[i+len_outage-1,'deficit']
        else:
            aggregated_deficit_prod.at[i,'deficit']=aggregated_deficit_prod.at[i-1,'deficit']-deficit_prod.at[i-1,'deficit']+deficit_prod.at[i+len_outage-deficit_prod.shape[0]-1,'deficit']
    aggregated_deficit_prod.rename(columns={"hour": "hour", "deficit": "aggregated_deficit_("+str(len_outage/24)+"d)"}, inplace=True)
    return aggregated_deficit_prod

def select_min(aggregated):
    test_aggregated = aggregated.copy()
    test_min = test_aggregated.idxmin(axis = 0)['aggregated_deficit_('+str(len_outage/24)+'d)']
    while impossible_values.at[test_min,'value'] == 0:
        test_aggregated.at[test_min,"aggregated_deficit_("+str(len_outage/24)+"d)"] += 300000 # Value is arbitrary
        test_min = test_aggregated.idxmin(axis = 0)['aggregated_deficit_('+str(len_outage/24)+'d)']
    return test_min

#Memoization
def update_impossible_values(min, operating_capacity):
    #This makes sure that a minimal share of the npp are available at all time
    for i in range(len_outage):
        h=min+i
        if h >= 8760*nb_years: h -= 8760*nb_years
        if operating_capacity.at[h,'capa'] - capa_per_plant < capa*min_share_online_npp: 
            impossible_values.at[h,'value'] = 0
    #This makes sure that plants do not stop at the same time due to human resources limitations
    for i in range(14*24):
        h=min+i-7*24
        if h >= 8760*nb_years: h -= 8760*nb_years
        if h < 0: h += 8760*nb_years
        if impossible_values.at[h,'value'] > 0: 
            impossible_values.at[h,'value'] -= 1

# For this function, we try every possible value for the first outage. 
# Values where deficit > 0 are excluded to reduce compute time for the 20y version. It is thought not to have an impact on the best solution given that deficit is similar every year.
# Then, we deduce other minimums with the step variable, while still allowing for variation with the delta parameter.
def select_mins(aggregated):
    test_aggregated = aggregated.copy()
    best_mins=[] # The best list of mins found so far
    best_value=3000000000000000000000 # The best deficit value found so far, used for comparaison
    for test_min in range(step): # We test every value for the first outage, using the minimum for the first year does not imply that the next ones will minimize the sum
        # Initialisation with first outage
        if impossible_values.at[test_min,'value'] == 0: # For Lynch constraints
            continue
        #if test_aggregated.at[test_min,"aggregated_deficit_("+str(len_outage/24)+"d)"] > 0: continue # To reduce compute time. Will not work properly when Lynch constraints are activated and installed capacity is large (> 10GW)
        test_min_with_delta = test_min
        value = test_aggregated.at[test_min,"aggregated_deficit_("+str(len_outage/24)+"d)"]
        for j in range(delta+1): # This apply the delta parameter to the first outage as well
            hour_plus_j = test_min + j if test_min + j < len(test_aggregated) else  test_min + j - len(test_aggregated)
            hour_minus_j = test_min - j if test_min - j >= 0 else test_min - j + len(test_aggregated)
            value_plus_j = test_aggregated.at[hour_plus_j,"aggregated_deficit_("+str(len_outage/24)+"d)"]
            value_minus_j = test_aggregated.at[hour_minus_j,"aggregated_deficit_("+str(len_outage/24)+"d)"]
            if value_minus_j < value:
                value = value_minus_j
                test_min_with_delta = hour_minus_j
            if value_plus_j:
                value = value_plus_j
                test_min_with_delta = hour_plus_j
        test_mins=[test_min_with_delta]
        previous_min_without_delta = test_min # We don't update this to test_min_with_delta, this is used as a reference to compute other mins for the delta parameter
        # Repeat initialisation for all other outages
        for i in range(nb_outages-1):
            value_year_i = 300000
            min_year_i = 0
            hour = previous_min_without_delta+step if previous_min_without_delta+step < len(test_aggregated) else previous_min_without_delta+step - len(test_aggregated)
            ### FOR LYNCH CONSTRAINTS
            is_valid = True
            is_valid_sum = 0 
            for j in range(delta+1):
                hour_plus_j = hour + j if hour + j < len(test_aggregated) else  hour + j - len(test_aggregated)
                hour_minus_j = hour - j if hour - j >= 0 else hour - j + len(test_aggregated)
                is_valid_sum += impossible_values.at[hour_minus_j,'value'] + impossible_values.at[hour_plus_j,'value']
            if is_valid_sum == 0: 
                is_valid = False
                break
            ###
            for j in range(delta+1):
                hour_plus_j = hour + j if hour + j < len(test_aggregated) else  hour + j - len(test_aggregated)
                hour_minus_j = hour - j if hour - j >= 0 else hour - j + len(test_aggregated)
                value_plus_j = test_aggregated.at[hour_plus_j,"aggregated_deficit_("+str(len_outage/24)+"d)"]
                value_minus_j = test_aggregated.at[hour_minus_j,"aggregated_deficit_("+str(len_outage/24)+"d)"]
                test_min_year_i = hour + j if value_plus_j < value_minus_j else hour - j
                if value_year_i > test_aggregated.at[test_min_year_i,"aggregated_deficit_("+str(len_outage/24)+"d)"]:
                    min_year_i = test_min_year_i
                    value_year_i = test_aggregated.at[test_min_year_i,"aggregated_deficit_("+str(len_outage/24)+"d)"]
            previous_min_without_delta += step
            value += value_year_i
            test_mins.append(int(min_year_i))
        if not is_valid:
            continue
        if value < best_value:
            best_value = value
            best_mins = test_mins
    if best_value == 3000000000000000000000:
        raise ValueError("There is an error in the minimum selection algorithm.")
    return best_mins

def rescale(deficit, capa, capa_min, min):
    # To understand the four distinct phases (full flex, limited flex, fixed output and outage), read A. Lynch et al. (2022)
    new_deficit=deficit.copy()
    new_capa=capa.copy()
    new_capa_min=capa_min.copy()
    # Outage : plant capacity set to 0
    for i in range(len_outage):
        hour=min+i # i=0 corresponds to the first hour
        if hour < deficit.shape[0]:
            new_deficit.at[hour, 'deficit']=deficit.at[hour, 'deficit'] + capa_per_plant
            new_capa.at[hour, 'capa']=capa.at[hour, 'capa'] - capa_per_plant
            new_capa_min.at[hour, 'min']=capa_min.at[hour, 'min'] - minimal_operating_power*capa_per_plant
        else:
            new_deficit.at[hour-deficit.shape[0], 'deficit']=deficit.at[hour-deficit.shape[0], 'deficit'] + capa_per_plant
            new_capa.at[hour-capa.shape[0], 'capa']=capa.at[hour-capa.shape[0], 'capa'] - capa_per_plant
            new_capa.at[hour-capa_min.shape[0], 'min']=capa_min.at[hour-capa_min.shape[0], 'min'] - minimal_operating_power*capa_per_plant
    # Fixed output : plant capacity is limited but known, ranging from 100% to 80% Pnom
    for i in range(len_fixed_output):
        hour=min+i-len_fixed_output # i=0 corresponds to the first hour
        if hour >= 0:
            new_deficit.at[hour, 'deficit']=deficit.at[hour, 'deficit'] + (1-power_outage_start)*capa_per_plant*i/(len_fixed_output-1)
            new_capa.at[hour, 'capa']=capa.at[hour, 'capa'] - (1-power_outage_start)*capa_per_plant*i/(len_fixed_output-1)
            new_capa_min.at[hour, 'min']=capa_min.at[hour, 'min'] + (-(1-power_outage_start)*i/(len_fixed_output-1) + 1 - minimal_operating_power)*capa_per_plant
        else:
            new_deficit.at[hour+deficit.shape[0], 'deficit']=deficit.at[hour+deficit.shape[0], 'deficit'] + (1-power_outage_start)*capa_per_plant*i/(len_fixed_output-1)
            new_capa.at[hour+capa.shape[0], 'capa']=capa.at[hour+capa.shape[0], 'capa'] - (1-power_outage_start)*capa_per_plant*i/(len_fixed_output-1)
            new_capa_min.at[hour+capa_min.shape[0], 'min']=capa_min.at[hour+capa_min.shape[0], 'min'] + (-(1-power_outage_start)*i/(len_fixed_output-1) + 1 - minimal_operating_power)*capa_per_plant
    # Limited flexibility : plant maximal capacity is unchanged but the minimal output gradually increases
    for i in range(len_limited_flexibility):
        hour=min+i-len_fixed_output-len_limited_flexibility # i=0 corresponds to the first hour
        if hour >= 0:
            new_capa_min.at[hour, 'min']=capa_min.at[hour, 'min'] + (1-minimal_operating_power)*i/(len_limited_flexibility-1)*capa_per_plant
        else:
            new_capa_min.at[hour+capa_min.shape[0], 'min']=capa_min.at[hour+capa_min.shape[0], 'min'] + (1-minimal_operating_power)*i/(len_limited_flexibility-1)*capa_per_plant
    # Full flexibility : flexibility ranges from 20% and 100% Pnom (already taken into account)
    return new_deficit, new_capa, new_capa_min

def step_by_step_plot(ini_aggregate, aggregate, mins, i, capa_max, capa_min): 
    # The evolution of the production deficit is not plotted because it is not very exploitable
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    aggregate.plot(ax=axes[0], x='hour', y='aggregated_deficit_('+str(len_outage/24)+'d)', label='demande', kind='line')
    ini_aggregate.plot(ax=axes[0], x='hour', y='aggregated_deficit_('+str(len_outage/24)+'d)', label='demande initiale', linestyle='--')
    for min in mins:
        axes[0].axvline(x=min)

    capa_max.plot(ax=axes[1], x='hour', y='capa', label='max')
    capa_min.plot(ax=axes[1], x='hour', y='min')
    plt.ylim([-0.1, 15.6])
    
    plt.setp(axes[0], ylabel='Demande résiduelle agrégée sur 30 jours (GWh)')
    plt.setp(axes[0], xlabel='Hours')
    plt.setp(axes[1], ylabel='Capacité disponible (GW)')
    plt.setp(axes[1], xlabel='Hours')
    fig.suptitle(str(i) + " outage(s) scheduled")
    plt.show()
    plt.clf()
    plt.close()

def main():
    nb_plant = int(np.floor(capa/capa_per_plant))
    deficit_prod = ini_deficit_prod
    schedule=[]
    operating_capacity=pd.DataFrame(data={'hour': [i for i in range(8760*nb_years)], 'capa': [nb_plant*capa_per_plant for i in range(8760*nb_years)]})
    operating_minimum=pd.DataFrame(data={'hour': [i for i in range(8760*nb_years)], 'min': [minimal_operating_power*nb_plant*capa_per_plant for i in range(8760*nb_years)]})
    ini_aggregate=aggregate(deficit_prod)
    aggregated=ini_aggregate
    for i in range(nb_plant):
        if nb_outages == 1: mins=[select_min(aggregated)]
        else:
            mins=select_mins(aggregated)
        schedule.append([str(date[min])+" (hour "+str(min)+")" for min in mins])
        if Step_by_step_plots and With_plots: step_by_step_plot(ini_aggregate, aggregated, mins, i, operating_capacity, operating_minimum)
        for min in mins:
            deficit_prod, operating_capacity, operating_minimum=rescale(deficit_prod, operating_capacity, operating_minimum, min)
        if With_Lynch_constraints: update_impossible_values(min, operating_capacity)
        aggregated=aggregate(deficit_prod)
        print("[" + str(int(100*(i+1)/nb_plant)) + "%] Outage(s) scheduled for NPP "+str(i+1))
        for min in range(len(mins)): 
            print("     - "+schedule[i][min])
    if With_plots:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
        aggregated.plot(ax=axes[0], x='hour', y='aggregated_deficit_('+str(len_outage/24)+'d)', kind='line')
        ini_aggregate.plot(ax=axes[0], x='hour', y='aggregated_deficit_('+str(len_outage/24)+'d)', linestyle='--')

        operating_capacity.plot(ax=axes[1], x='hour', y='capa', label='max')
        operating_minimum.plot(ax=axes[1], x='hour', y='min')
        plt.ylim([-0.1, 15.6])
        
        plt.setp(axes[0], ylabel='Demande résiduelle agrégée sur 30 jours (GWh)')
        plt.setp(axes[0], xlabel='Hours')
        plt.setp(axes[1], ylabel='Capacité disponible (GW)')
        plt.setp(axes[1], xlabel='Hours')
        fig.suptitle(str(int(np.floor(capa/capa_per_plant)))+" outage(s) scheduled")
        plt.show()
        plt.clf()
        plt.close()
    return operating_capacity, operating_minimum

operating_capacity, operating_minimum = main()

### OUTPUTS ###

in_fix_capa = pd.read_csv("inputs/fix_capa.csv", header=None)
add=pd.DataFrame([['nuc',round(int(np.floor(capa/capa_per_plant))*capa_per_plant,4)]],columns=[0,1])
out_fix_capa = pd.concat([in_fix_capa, add])
out_fix_capa.to_csv(path_or_buf='./inputs/fix_capa.csv', index=False, header= False)

operating_capacity.to_csv(path_or_buf='./inputs/nuclear/maximal_operating_capacity_'+str(nb_years)+'y.csv', header= False, index=False)

operating_minimum.to_csv(path_or_buf='./inputs/nuclear/minimal_operating_capacity_'+str(nb_years)+'y.csv', header= False, index=False)

print("Execution time (nuc opti module): " + str(round(time.time() - start, 2)) + "s")