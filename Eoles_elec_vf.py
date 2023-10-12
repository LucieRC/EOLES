# -*- coding: utf-8 -*-

"""
Eoles Model from Behrang Shirizadeh, Quentin Perrier and Philippe Quirion, May 2021
Written in Python by Nilam De Oliveira-Gill, June 2021
Overall improvements by Quentin Bustarret and Laure Baratgin, May 2022 - February 2023
"""

"""IMPORTS

Import modules and libraries needed for the programm 
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import csv
import time
import sys
from CONFIG import model_name, nb_years, input_years, demand_profile

#Initialize time to measure the execution time
start_time = time.time()

"""INITIALISATION OF THE MODEL"""

model = pyo.ConcreteModel()

#Dual Variable, used to get the marginal value of an equation.
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

"""INPUTS"""

    # Production profiles of VRE (capacity factors)
load_factor = pd.read_csv("inputs/vre_profiles_19y.csv")
    # Demand profile at each hour in GWh
demand_1y = pd.read_csv("inputs/demand"+demand_profile+".csv", index_col=0, header=None).squeeze("columns")
    # Daily lake & reservoir inflows in GWh
lake_inflows_daily = pd.read_csv("inputs/hydro/lake_inflows_19y.csv")
    # Daily phs inflows from precipitation and glacier melting, in GWh
phs_inflows_daily = pd.read_csv("inputs/hydro/phs_inflows_19y.csv")
    # Fixed energy capacity for each regional dam or reservoir in GWh
lake_capacity = pd.read_csv("inputs/hydro/lake_capacity.csv", index_col=0, header=None).squeeze("columns")
    # Fixed power capacity for each regional dam or reservoir in GW
lake_capa = pd.read_csv("inputs/hydro/lake_capa.csv", index_col=0, header=None).squeeze("columns")
    # Minimal peak energy volume that must be kept for non-electric purposes (ex. recreational) for each regional dam or reservoir in GWh. Exact profile is editable within the code
lake_peak_minimal_volume = pd.read_csv("inputs/hydro/lake_minimal_volume.csv", index_col=0, header=None).squeeze("columns")
    # Daily lake production spillage or each regional dam or reservoir in GWh. This corresponds to daily inflow that must be, otherwise lost. This is a correction to get closer to a real situation with much more dams, computed when constructing input data.
lake_production_spillage_daily = pd.read_csv("inputs/hydro/lake_prod_spill_19y.csv")
    # Daily lake maximal capa for each regional dam or reservoir in GW. This corresponds to the maximal capacity, minus the capacity that must be allocated to the production spillage. This is a correction to get closer to a real situation with much more dams, computed when constructing input data.
lake_max_capa_daily = pd.read_csv("inputs/hydro/lake_max_capa_19y.csv")
    # Additional FRR requirement for variable renewable energies because of forecast errors
epsilon = pd.read_csv("inputs/reserve_requirements_new.csv", index_col=0, header=None).squeeze("columns")
    # Existing capacities of the technologies by December 2017 in GW
existing_capa = pd.read_csv("inputs/existing_capa.csv", index_col=0, header=None).squeeze("columns")
    # Existing storage capacity in GWh
existing_capacity = pd.read_csv("inputs/existing_capacity.csv", index_col=0, header=None).squeeze("columns")
    # Maximum capacities of the technologies in GW
max_capa = pd.read_csv("inputs/max_capa.csv", index_col=0, header=None).squeeze("columns")
    # Maximum storage capacities of the technologies in GWh
max_capacity = pd.read_csv("inputs/max_capacity.csv", index_col=0, header=None).squeeze("columns")
    # Maximum allowed yearly production of the technologies in GWh
max_production = pd.read_csv("inputs/max_production.csv", index_col=0, header=None).squeeze("columns")
    # Fixed capacities of the technologies in GW
fix_capa = pd.read_csv("inputs/fix_capa.csv", index_col=0, header=None).squeeze("columns")
    # Fixed storage capacities of the technologies in GWh
fix_capacity = pd.read_csv("inputs/fix_capacity.csv", index_col=0, header=None).squeeze("columns")
    # Fixed charging capacities of the technologies in GW
fix_charging = pd.read_csv("inputs/fix_charging.csv", index_col=0, header=None).squeeze("columns")
    # Expected lifetime in years
lifetime = pd.read_csv("inputs/lifetime.csv", index_col=0, header=None).squeeze("columns")
    # Construction time in years
construction_time = pd.read_csv("inputs/construction_time.csv", index_col=0, header=None).squeeze("columns")
    # Overnight capex cost in M€/GW
capex = pd.read_csv("inputs/overnight_capex.csv", index_col=0, header=None).squeeze("columns")    
    # Energy capex cost fo storage technologies in M€/GWh
storage_capex = pd.read_csv("inputs/storage_capex.csv", index_col=0, header=None).squeeze("columns")
    # Annualized fixed operation and maintenance costs M€/GW/year
fOM = pd.read_csv("inputs/fO&M.csv", index_col=0, header=None).squeeze("columns")
    # Variable operation and maintenance costs in M€/GWh
vOM = pd.read_csv("inputs/vO&M.csv", index_col=0, header=None).squeeze("columns")
    # Charging related annuity of storage in M€/GW/year
charging_capex = pd.read_csv("inputs/charging_capex.csv", index_col=0, header=None).squeeze("columns")
    # Charging related fOM of storage in M€/GW/year
charging_opex = pd.read_csv("inputs/charging_opex.csv", index_col=0, header=None).squeeze("columns")
    # Charging efficiency of storage technologies
eta_in = pd.read_csv("inputs/eta_in.csv", index_col=0, header=None).squeeze("columns")
    # Production efficiency of technologies
eta_out = pd.read_csv("inputs/eta_out.csv", index_col=0, header=None).squeeze("columns")
    # The maximal operating capacity due to nuclear planned outages in GW
nuc_max_operating_capa = pd.read_csv("inputs/nuclear/maximal_operating_capacity_"+str(nb_years)+"y.csv", index_col=0, header=None).squeeze("columns")
    # The minimal operating capacity due to nuclear planned outages in GW
nuc_min_operating_capa = pd.read_csv("inputs/nuclear/minimal_operating_capacity_"+str(nb_years)+"y.csv", index_col=0, header=None).squeeze("columns")
    # Miscellaneous
miscellaneous = pd.read_csv("inputs/miscellaneous.csv", index_col=0, header=None).squeeze("columns")

"""Miscellaneous

Additional modifications on inputs data"""

    # Annualized power capex cost for each tec in M€/GW/year. Formula found in "Low-carbon options for the French power sector: What role for renewables, nuclear energy and carbon capture and storage?", Behrang Shirizadeh & Philippe Quirion, Energy Economics
annuities = pd.read_csv("inputs/overnight_capex.csv", index_col=0, header=None).squeeze("columns")
for i in annuities.index:
    annuities.at[i] = miscellaneous["discount_rate"]*capex[i]*(miscellaneous["discount_rate"]*construction_time[i]+1)/(1-(1+miscellaneous["discount_rate"])**(-lifetime[i]))

    # Annualized energy capex cost of storage technologies in M€/GWh/year   
storage_annuities = pd.read_csv("inputs/storage_capex.csv", index_col=0, header=None).squeeze("columns")
for i in storage_annuities.index:
    storage_annuities.at[i] = miscellaneous["discount_rate"]*storage_capex[i]*(miscellaneous["discount_rate"]*construction_time[i]+1)/(1-(1+miscellaneous["discount_rate"])**(-lifetime[i]))

    # Duplication of demand 
demand=demand_1y
for i in range(nb_years-1):
    demand = pd.concat([demand, demand_1y], ignore_index=True) 

    # Selection of requested years in the vre profiles
if len(input_years) < nb_years:
    print("Warning : "+str(nb_years)+" years were requested but only "+str(len(input_years))+" years of vre production profiles were inputed in the CONFIG.py file. Only "+str(len(input_years))+" years are being used.")
    nb_years = len(input_years)
load_factor['hour']=load_factor['hour'].apply(lambda x: x + nb_years*8760) # Shifting all data
for i in range(nb_years):
    year = int(input_years[i]) - 2000 # year 20xx has index xx 
    # Selection of year data, and shifting back the selected year
    load_factor['hour']=load_factor['hour'].apply(lambda x: x - (nb_years+year-i)*8760 if x >= 8760*(nb_years+year) and x < 8760*(nb_years+year+1) else x) 
load_factor = load_factor[load_factor.hour < nb_years*8760] # Filtering out remaining years of unused data
# Exporting result as pandas Series
tecs=load_factor.iloc[:,0].values
hours=load_factor.iloc[:,1].values
value=load_factor.iloc[:,2].values
load_factor = pd.Series(value, index=[tecs, hours])
# Overide with old values :
#load_factor = pd.read_csv("inputs/vre_profiles_2006_old.csv", index_col=[0, 1], header=None).squeeze("columns") 

    # Selection of requested years in the lake inflows
lake_inflows_daily['day']=lake_inflows_daily['day'].apply(lambda x: x + nb_years*365) # Shifting all data
for i in range(nb_years):
    year = int(input_years[i]) - 2000 # year 20xx has index xx 
    # Selection of year data, and shifting back the selected year
    lake_inflows_daily['day']=lake_inflows_daily['day'].apply(lambda x: x - (nb_years+year-i)*365 if x >= 365*(nb_years+year) and x < 365*(nb_years+year+1) else x) 
lake_inflows_daily = lake_inflows_daily[lake_inflows_daily.day < nb_years*365] # Filtering out remaining years of unused data
# Exporting result as pandas Series
tecs=lake_inflows_daily.iloc[:,0].values
days=lake_inflows_daily.iloc[:,1].values
value=lake_inflows_daily.iloc[:,2].values
lake_inflows_daily = pd.Series(value, index=[tecs, days])

    # Selection of requested years in the lake inflows
phs_inflows_daily['day']=phs_inflows_daily['day'].apply(lambda x: x + nb_years*365) # Shifting all data
for i in range(nb_years):
    year = int(input_years[i]) - 2000 # year 20xx has index xx 
    # Selection of year data, and shifting back the selected year
    phs_inflows_daily['day']=phs_inflows_daily['day'].apply(lambda x: x - (nb_years+year-i)*365 if x >= 365*(nb_years+year) and x < 365*(nb_years+year+1) else x) 
phs_inflows_daily = phs_inflows_daily[phs_inflows_daily.day < nb_years*365] # Filtering out remaining years of unused data
# Exporting result as pandas Series
days=phs_inflows_daily.iloc[:,0].values
value=phs_inflows_daily.iloc[:,1].values
phs_inflows_daily = pd.Series(value, index=[days])

    # Selection of requested years in the lake prod spill
lake_production_spillage_daily['day']=lake_production_spillage_daily['day'].apply(lambda x: x + nb_years*365) # Shifting all data
for i in range(nb_years):
    year = int(input_years[i]) - 2000 # year 20xx has index xx 
    # Selection of year data, and shifting back the selected year
    lake_production_spillage_daily['day']=lake_production_spillage_daily['day'].apply(lambda x: x - (nb_years+year-i)*365 if x >= 365*(nb_years+year) and x < 365*(nb_years+year+1) else x) 
lake_production_spillage_daily = lake_production_spillage_daily[lake_production_spillage_daily.day < nb_years*365] # Filtering out remaining years of unused data
# Exporting result as pandas Series
tecs=lake_production_spillage_daily.iloc[:,0].values
days=lake_production_spillage_daily.iloc[:,1].values
value=lake_production_spillage_daily.iloc[:,2].values
lake_production_spillage_daily = pd.Series(value, index=[tecs, days])

    # Selection of requested years in the lake max capa
lake_max_capa_daily['day']=lake_max_capa_daily['day'].apply(lambda x: x + nb_years*365) # Shifting all data
for i in range(nb_years):
    year = int(input_years[i]) - 2000 # year 20xx has index xx 
    # Selection of year data, and shifting back the selected year
    lake_max_capa_daily['day']=lake_max_capa_daily['day'].apply(lambda x: x - (nb_years+year-i)*365 if x >= 365*(nb_years+year) and x < 365*(nb_years+year+1) else x) 
lake_max_capa_daily = lake_max_capa_daily[lake_max_capa_daily.day < nb_years*365] # Filtering out remaining years of unused data
# Exporting result as pandas Series
tecs=lake_max_capa_daily.iloc[:,0].values
days=lake_max_capa_daily.iloc[:,1].values
value=lake_max_capa_daily.iloc[:,2].values
lake_max_capa_daily = pd.Series(value, index=[tecs, days])

"""SETS

Definition of set as an object of the model
"""

#Range of hour
model.h = pyo.RangeSet(0, len(demand)-1)
#Months
model.months = pyo.RangeSet(0, 12*nb_years-1)
#Years
model.years = pyo.RangeSet(0,nb_years-1)
#Technologies
model.tec = \
    pyo.Set(initialize=["offshore_f", "offshore_g", "onshore_V90", "onshore_V110", "pv_g_EW", "pv_g_S", "pv_c", "river", "lake", "small_hydro", "methanization", "ocgt", "ccgt", "nuc", "h2_ccgt", "phs",  "battery1", "battery4", "methanation", "pyrogazification", "electrolysis", "hydrogen", "methane", "waste", "geothermal_coge", "biomass_coge"])
#Power plants. Only used to calculate sum of generation, excluding storage.
model.gen = \
    pyo.Set(initialize=["offshore_f", "offshore_g", "onshore_V90", "onshore_V110", "pv_g_EW", "pv_g_S", "pv_c", "river", "lake", "small_hydro", "ocgt", "ccgt", "nuc", "h2_ccgt", "waste", "geothermal_coge", "biomass_coge"])
#Variables Technologies
model.vre = \
    pyo.Set(initialize=["offshore_f", "offshore_g", "onshore_V90", "onshore_V110", "pv_g_EW", "pv_g_S", "pv_c", "river", "small_hydro", "waste", "geothermal_coge", "biomass_coge"])
#Biogas technologies
model.biogas = \
    pyo.Set(initialize=["methanization", "methanation", "pyrogazification", "methane"])
#Hydrogen technologies
model.hydrogen = \
    pyo.Set(initialize=["electrolysis", "hydrogen"])
#Electricity generating technologies
model.balance = \
    pyo.Set(initialize=["offshore_f", "offshore_g", "onshore_V90", "onshore_V110", "pv_g_EW", "pv_g_S", "pv_c", "river", "lake", "small_hydro", "ocgt", "ccgt", "nuc", "h2_ccgt", "phs", "battery1", "battery4", "waste", "geothermal_coge", "biomass_coge"])
#Storage Technologies
model.str = \
    pyo.Set(initialize=["phs", "battery1", "battery4", "hydrogen", "methane"])
#Battery Storage
model.battery = \
    pyo.Set(initialize=["battery1", "battery4"])
#Technologies for upward FRR
model.frr = \
    pyo.Set(initialize=["lake", "phs", "ocgt", "ccgt", "nuc", "h2_ccgt"])
#Dams & Reservoirs
model.lakes = \
    pyo.Set(initialize=["ALL_DAM", "ALL_ECLUS"])
#Dams
model.dams = \
    pyo.Set(initialize=["ALL_DAM"])
#Reservoirs (locks)
model.reservoirs = \
    pyo.Set(initialize=["ALL_ECLUS"])

#Dictionnaries which will set a little definition and unit for each technology in the model.
technologies_definition = {
    "offshore_f" : "floating offshore wind",
    "offshore_g" : "ground-based offshore wind",
    "onshore_V90" : "onshore wind, technology V90", 
    "onshore_V110" : "onshore wind, technology V110",
    "pv_g_EW" : "pv grounded, oriented East-West",
    "pv_g_S" : "pv grounded, oriented South",
    "pv_c" : "pv commercial",
    "river" : "run-of-river hydro",
    "lake" : "lake and reservoirs",
    "small_hydro" : "minor hydro tecs",
    "methanization" : "biogas from methanization",
    "ocgt" : "open cycle gas-turbine",
    "ccgt" : "combined cycle gas turbine",
    "nuc" : "nuclear",
    "h2_ccgt" : "combined cycle gas turbine using hydrogen",
    "phs" : "pumped hydroelectric energy storage",
    "battery1" : "1 hour battery",
    "battery4" : "4 hours battery",
    "methanation" : "methane by methanation, into storage",
    "pyrogazification" : "methane by pyrogazification, into storage",
    "electrolysis" : "electrolysis",
    "hydrogen" : "hydrogen storage in salt caverns",
    "methane" : "methane storage",
    "waste" : "cogeneration from waste disposal",
    "biomass_coge" : "cogeneration from solid biomass",
    "geothermal_coge" : "cogeneration from geothermal"
}

technologies_units = {
    "offshore_f" : "GWh-e",
    "offshore_g" : "GWh-e",
    "onshore_V90": "GWh-e",
    "onshore_V110" : "GWh-e",
    "pv_g_EW" : "GWh-e",
    "pv_g_S" : "GWh-e",
    "pv_c" : "GWh-e",
    "river" : "GWh-e",
    "lake" : "GWh-e",
    "small_hydro" : "GWh-e",
    "methanization" : "GWh-th",
    "ocgt" : "GWh-e",
    "ccgt" : "GWh-e",
    "nuc" : "GWh-e",
    "h2_ccgt" : "GWh-e",
    "phs" : "GWh-e",
    "battery1" : "GWh-e",
    "battery4" : "GWh-e",
    "methanation" : "GWh-th",
    "pyrogazification" : "GWh-th",
    "electrolysis" : "GWh-th",
    "hydrogen" : "GWh-th",
    "methane" : "GWh-th",
    "waste" : "GWh-e",
    "biomass_coge" : "GWh-e",
    "geothermal_coge" : "GWh-e"
}

"""PARAMETERS"""

# Changes daily inflows (best resolution for input data) to homogen hourly inflows (resolution of the model)
lake_inflows = {}
for lake in model.lakes:
    for h in model.h:
        day = int(h/24)
        lake_inflows[lake, h] = lake_inflows_daily[lake, day]/24
lake_inflows = pd.Series(list(lake_inflows.values()),index=pd.MultiIndex.from_tuples(lake_inflows.keys()))

# Imports phs inflows into a general parameter for all storage tecs. You can use this for other storage tecs with external inputs
storage_inflow = {}
for str_tec in model.str:
    for h in model.h:
        if str_tec != "phs":
            storage_inflow[str_tec, h] = 0
        else:
            day = int(h/24)
            storage_inflow[str_tec, h] = phs_inflows_daily[day]/24
storage_inflow = pd.Series(list(storage_inflow.values()),index=pd.MultiIndex.from_tuples(storage_inflow.keys()))

# Sets the profile for lake volume reserved for recreational purposes : ramping from a to b, fix from b to c, 0 otherwise.
lake_minimal_capacity = {}
a, b, c = 2400, 4200, 5600
for lake in model.lakes:
    for h in model.h:
        hour = h % 8760
        if hour <= a or hour > c:
            lake_minimal_capacity[lake, h] = 0
        elif hour <= b:
            lake_minimal_capacity[lake, h] = 0.9 * lake_peak_minimal_volume[lake] * (hour - a)/(b-a)
        else: # b < h <= c
            lake_minimal_capacity[lake, h] = 0.9 * lake_peak_minimal_volume[lake]
lake_minimal_capacity = pd.Series(list(lake_minimal_capacity.values()),index=pd.MultiIndex.from_tuples(lake_minimal_capacity.keys()))

# Changes daily potential spillage to hourly timestep
lake_prod_to_be_spilled = {}
for lake in model.lakes:
    for h in model.h:
        day = int(h/24)
        lake_prod_to_be_spilled[lake, h] = lake_production_spillage_daily[lake, day]/24
lake_prod_to_be_spilled = pd.Series(list(lake_prod_to_be_spilled.values()),index=pd.MultiIndex.from_tuples(lake_prod_to_be_spilled.keys()))

# Changes daily max capa to hourly timestep
lake_max_capa = {}
for lake in model.lakes:
    for h in model.h:
        day = int(h/24)
        lake_max_capa[lake, h] = lake_max_capa_daily[lake, day]
lake_max_capa = pd.Series(list(lake_max_capa.values()),index=pd.MultiIndex.from_tuples(lake_max_capa.keys()))

"""BOUNDS VALUES

Set initial value for variables.
There is a function for each variable with bounds.
The function return the lower and the upper value.
"""

def capa_bounds(model,i):
    min = None
    max = None
    if i in existing_capa.keys():
        min = existing_capa[i]
    if i in max_capa.keys():
        max = max_capa[i]
    return (min,max)

def capacity_bounds(model,i):
    min = None
    max = None
    if i in existing_capacity.keys():
        min = existing_capacity[i]
    if i in max_capacity.keys():
        max = max_capacity[i]
    return (min,max)

"""VARIABLES

Definition of variable as an object of the model
"""

    # Hourly energy generation in GWh/h
model.gene = \
    pyo.Var(((tec, h) for tec in model.tec for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Overall yearly installed capacity in GW
model.capa = \
    pyo.Var(model.tec, within=pyo.NonNegativeReals, bounds=capa_bounds)

    # Hourly electricity input of battery storage GW
model.storage = \
    pyo.Var(((storage, h) for storage in model.str for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Energy stored in each storage technology in GWh = Stage of charge
model.stored = \
    pyo.Var(((storage, h) for storage in model.str for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Charging power capacity of each storage technology
model.s = \
    pyo.Var(model.str, within=pyo.NonNegativeReals, initialize=0)

    # Energy volume of storage technology in GWh
model.capacity = \
    pyo.Var(model.str, within=pyo.NonNegativeReals, bounds=capacity_bounds)

    # Energy stored in each dam or reservoir in GWh
model.lake_stored = \
    pyo.Var(((lake, h) for lake in model.lakes for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Individual hourly generation of each dam or reservoir
model.lake_gene = \
    pyo.Var(((lake, h) for lake in model.lakes for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Inflow spilled due to inflow greater than capa at a given hour with a full stock
model.lake_spilled = \
    pyo.Var(((lake, h) for lake in model.lakes for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # Required upward frequency restoration reserve in GW    
model.reserve = \
    pyo.Var(((reserve, h) for reserve in model.frr for h in model.h), within=pyo.NonNegativeReals,initialize=0)

    # Hourly unavailable nuclear capacity
try:
    model.capa_nuc_off = \
        pyo.Var(model.h, within=pyo.NonNegativeReals, initialize=0, bounds=(0, fix_capa['nuc']))
except Exception as err:
    raise ValueError("For running the EOLES model with the corrected nuclear flexibility constraints, run the RUN.py file with pyomo solve instead, or fix the installed nuclear capacity in ./inputs/fix_capa.csv. If you want to run a simplified version of EOLES, run Eoles_elec_vf_preprocess instead.")

"""FIXED VALUES"""

for tec in model.tec:
    if tec in fix_capa.keys():
        model.capa[tec].fix(fix_capa[tec])
    if tec in fix_capacity.keys():
        model.capacity[tec].fix(fix_capacity[tec])
    if tec in fix_charging.keys():
        model.s[tec].fix(fix_charging[tec])

"""CONSTRAINTS RULE

Set up a function which will return the equation of the constraint.
"""

###############
### GENERAL ###
###############

def generation_capacity_constraint_rule(model, h, tec):
    """Get constraint on maximum power for non-VRE technologies."""

    return model.capa[tec] >= model.gene[tec,h]
model.generation_capacity_constraint = pyo.Constraint(model.h, model.tec, rule=generation_capacity_constraint_rule)

def frr_capacity_constraint_rule(model, h, frr):
    """Get constraint on maximum generation including reserves"""

    return model.capa[frr] >= model.gene[frr, h] + model.reserve[frr, h]
model.frr_capacity_constraint = pyo.Constraint(model.h, model.frr, rule=frr_capacity_constraint_rule)

def reserves_constraint_rule(model, h):
    """Get constraint on frr reserves"""

    res_req = sum(epsilon[vre] * model.capa[vre] for vre in model.vre)
    load_req = demand[h] * miscellaneous['load_uncertainty'] * (1 + miscellaneous['delta'])
    return sum(model.reserve[frr, h] for frr in model.frr) ==  res_req + load_req
model.reserves_constraint = pyo.Constraint(model.h, rule=reserves_constraint_rule)

def adequacy_constraint_rule(model, h):
    """Get constraint for 'supply/demand relation'"""

    storage = sum(model.storage[str, h] for str in model.str if (str != "hydrogen" and str != "methane"))
    gene_electrolysis = model.gene['electrolysis',h] / eta_out['electrolysis']
    gene_methanation = model.gene['methanation',h] / eta_out['methanation']
    return sum(model.gene[balance, h] for balance in model.balance) >= (demand[h] + storage + gene_electrolysis + gene_methanation)
model.adequacy_constraint = pyo.Constraint(model.h, rule=adequacy_constraint_rule)
# NB : this implies that demand meets consumption at each hour.
# This might not be the case if you restrict installed capacities or increase demand too much
# If you do so (to verify that a given scenario is plausible for instance), you need to add a variable that mesures unsatisfied demand
# Then, add it to the objective rule with a cost (cost not te meet demand, or a very high cost if you what as little unsatisfied demand as possible)
# and modify the adequacy constraint like so : sum(model.gene[balance, h] for balance in model.balance) + model.unsatisfied_demand[h] >= ...

def objective_rule(model):
    """Get constraint for the final objective function"""

    return (sum((model.capa[tec] - existing_capa[tec]) * annuities[tec] * nb_years for tec in model.tec) \
           + sum((model.capacity[storage_tecs]-existing_capacity[storage_tecs]) * storage_annuities[storage_tecs] * nb_years for storage_tecs in model.str) \
           + sum(model.capa[tec] * fOM[tec] * nb_years for tec in model.tec) \
           + sum(model.s[storage_tecs] * (charging_opex[storage_tecs] + charging_capex[storage_tecs]) * nb_years for storage_tecs in model.str) \
           + sum(sum(model.gene[tec, h] * vOM[tec] for h in model.h) for tec in model.tec) \
           )/1000
#Creation of the objective -> Cost
model.objective = pyo.Objective(rule=objective_rule)

###########
### VRE ###
###########

def onshore_capa_bound_constraint_rule(model):
    """Set capacity bound for all onshore wind technologies"""

    return model.capa["onshore_V90"] + model.capa["onshore_V110"] <= max_capa["onshore"]
model.onshore_capa_bound_constraint = pyo.Constraint(rule=onshore_capa_bound_constraint_rule)

def pv_g_capa_bound_constraint_rule(model):
    """Set capacity bound for all grounded PV technologies"""

    return model.capa["pv_g_EW"] + model.capa["pv_g_S"] <= max_capa["pv_g"]
model.pv_g_capa_bound_constraint = pyo.Constraint(rule=pv_g_capa_bound_constraint_rule)

def generation_vre_constraint_rule(model, h, vre):
    """Get constraint on variables renewable profiles generation."""

    return model.gene[vre, h] == model.capa[vre] * load_factor[vre,h]
model.generation_vre_constraint = pyo.Constraint(model.h, model.vre, rule=generation_vre_constraint_rule)

###############
### STORAGE ###
###############

def battery1_capacity_constraint_rule(model):
    """Get constraint on capacity of battery1."""

    return model.capa['battery1'] == model.capacity['battery1']
model.battery_1_capacity_constraint = pyo.Constraint(rule=battery1_capacity_constraint_rule)

def battery4_capacity_constraint_rule(model):
    """Get constraint on capacity of battery4."""

    return model.capa['battery4'] == model.capacity['battery4']/4
model.battery_4_capacity_constraint = pyo.Constraint(rule=battery4_capacity_constraint_rule)

def stored_capacity_constraint_rule(model, h, storage_tecs):
    """Get constraint on maximum energy that is stored in storage units"""

    return model.stored[storage_tecs, h] <= model.capacity[storage_tecs]
model.stored_capacity_constraint = pyo.Constraint(model.h, model.str, rule=stored_capacity_constraint_rule)

def storage_capacity_constraint_rule(model, h, storage_tecs):
    """Get constraint on the capacity with hourly charging relationship of storage"""

    return model.storage[storage_tecs, h] <= model.s[storage_tecs]
model.storage_capacity_constraint = pyo.Constraint(model.h, model.str, rule=storage_capacity_constraint_rule)

def battery_capacity_constraint_rule(model, battery):
    """Get constraint on battery's capacity."""

    return model.s[battery] == model.capa[battery]
model.battery_capacity_constraint = pyo.Constraint(model.battery, rule=battery_capacity_constraint_rule)

def storing_constraint_rule(model, h, storage_tec):
    """Get constraint on storing."""

    hPOne = h+1 if h < model.h.last() else 0
    charge = model.storage[storage_tec, h] * eta_in[storage_tec]
    discharge = model.gene[storage_tec, h] / eta_out[storage_tec]
    flux = charge - discharge + storage_inflow[storage_tec, h]
    return model.stored[storage_tec, hPOne] == model.stored[storage_tec, h] + flux 
model.storing_constraint = pyo.Constraint(model.h, model.str, rule=storing_constraint_rule)

##############
### BIOGAS ###
##############

def biogas_methanization_constraint_rule(model,year):
    """Get constraint on methanization."""

    return sum(model.gene['methanization',h] for h in range(8760*year,8760*(year+1)-1)) <= max_production['methanization']
model.biogas_methanization_constraint = pyo.Constraint(model.years,rule=biogas_methanization_constraint_rule)

def biogas_pyro_constraint_rule(model,year):
    """Get constraint on pyrogazification."""

    return sum(model.gene['pyrogazification',h] for h in range(8760*year,8760*(year+1)-1)) <= max_production['pyrogazification']
model.biogas_pyro_constraint = pyo.Constraint(model.years,rule=biogas_pyro_constraint_rule)

def methanation_constraint_rule(model):
    """Get constraint on CO2's balance from methanization"""

    return sum(model.gene['methanation',h] for h in model.h) <= sum(model.gene['methanization',h] for h in model.h) * miscellaneous['percentage_co2_from_methanization']
model.methanation_constraint = pyo.Constraint(rule=methanation_constraint_rule)

def methane_storage_constraint_rule(model,h):
    """Every biogas produced is stored..."""

    return model.storage['methane', h] == model.gene['methanation',h] + model.gene['methanization',h] + model.gene['pyrogazification', h]
model.methane_storage_constraint = pyo.Constraint(model.h,rule=methane_storage_constraint_rule)

def methane_balance_constraint_rule(model,h):
    """... and then distributed"""

    return model.gene['methane',h] == model.gene['ocgt',h]/eta_out['ocgt'] + model.gene['ccgt',h]/eta_out['ccgt'] + miscellaneous['CH4_demand']/8760
model.methane_balance_constraint = pyo.Constraint(model.h,rule=methane_balance_constraint_rule)

################
### HYDROGEN ###
################

def hydrogen_storage_constraint_rule(model, h):
    """Electrolysis production goes straight into storage..."""

    return model.gene['electrolysis',h] == model.storage['hydrogen',h]
model.hydrogen_storage_constraint = pyo.Constraint(model.h,rule=hydrogen_storage_constraint_rule)

def hydrogen_balance_constraint_rule(model,h):
    """... which is then distibuted"""

    return model.gene['hydrogen',h] >= model.gene['h2_ccgt',h]/eta_out['h2_ccgt']
model.hydrogen_balance_constraint = pyo.Constraint(model.h,rule=hydrogen_balance_constraint_rule)

def hydrogen_demand_constraint_rule(model, month):
    """Every H2 not used in H2_ccgt satisfies a monthly hydrogen demand"""

    return sum(model.gene['hydrogen',h] - model.gene['h2_ccgt',h]/eta_out['h2_ccgt'] for h in range(month*730,(month+1)*730)) == miscellaneous['H2_demand'] / 12
model.hydrogen_demand_constraint = pyo.Constraint(model.months,rule=hydrogen_demand_constraint_rule)

###############
### NUCLEAR ###
###############

def ramping_nuc_up_constraint_rule(model, h):
    """Sets an upper ramping limit for nuclear flexibility"""

    old_h = model.h.last() if h==0 else h-1
    return model.gene['nuc',h] - model.gene['nuc',old_h] + model.reserve['nuc',h] - model.reserve['nuc',old_h] <= miscellaneous['hourly_ramping_nuc']*model.capa['nuc']
model.ramping_nuc_up_constraint = pyo.Constraint(model.h, rule=ramping_nuc_up_constraint_rule)

def ramping_nuc_down_constraint_rule(model, h):
    """Sets a lower ramping limit for nuclear flexibility"""

    old_h = model.h.last() if h==0 else h-1
    return model.gene['nuc',old_h] - model.gene['nuc',h] + model.reserve['nuc',old_h] - model.reserve['nuc',h] <= miscellaneous['hourly_ramping_nuc']*model.capa['nuc']
model.ramping_nuc_down_constraint = pyo.Constraint(model.h, rule=ramping_nuc_down_constraint_rule)

def max_nuc_power_constraint_rule(model, h):
    """Takes into account the reactors stopped for maintenance / refuelling while still allowing for shutdowns during the irradiation period"""

    return model.gene['nuc', h] + model.reserve['nuc', h] <= nuc_max_operating_capa[h] - model.capa_nuc_off[h]
model.max_nuc_power_constraint = pyo.Constraint(model.h, rule=max_nuc_power_constraint_rule)

def min_nuc_power_constraint_rule(model, h):
    """Takes into account the reactors stopped for maintenance / refuelling while still allowing for shutdowns during the irradiation period"""

    return model.gene['nuc', h] + model.reserve['nuc', h] >= nuc_min_operating_capa[h] - 0.25*model.capa_nuc_off[h]
model.min_nuc_power_constraint = pyo.Constraint(model.h, rule=min_nuc_power_constraint_rule)

###
# This part is purely to replicate a function which is not supported by pyomo.
# The model require a variable model.positive_value which equals model.capa_nuc_off[h]-model.capa_nuc_off[h-1] if positive else 0

model.helper = pyo.Var(model.h, within=pyo.Reals, initialize=0)
model.helper_positive = pyo.Var(model.h, within=pyo.NonNegativeReals, initialize=0)
model.helper_negative = pyo.Var(model.h, within=pyo.NonPositiveReals, initialize=0)

def helper_abs_rule(model, h):
    return model.helper[h] == model.helper_positive[h] + model.helper_negative[h]

def helper_no_cheat_rule(model, h):
    return model.helper_positive[h] <= model.capa_nuc_off[h]

def helper_def_rule(model, h):
    h_old = model.h.last() if h == 0 else h-1 
    return model.helper[h] == model.capa_nuc_off[h] - model.capa_nuc_off[h_old]

model.helper_no_cheat = pyo.Constraint(model.h, rule=helper_no_cheat_rule)
model.helper_abs = pyo.Constraint(model.h, rule=helper_abs_rule)
model.helper_def = pyo.Constraint(model.h, rule=helper_def_rule)
###

def delay_nuc_power_constraint_rule(model, h):
    """Allows for shutdowns during the irradiation period but introduces a delay for next startup due to xenon transient"""

    h_delay = h-miscellaneous['len_delay'] if h >= miscellaneous['len_delay'] else h-miscellaneous['len_delay']+len(demand)
    h_old = model.h.last() if h == 0 else h-1
    return model.capa_nuc_off[h] >= model.capa_nuc_off[h_old] - model.helper_positive[h_delay]
model.delay_nuc_power_constraint = pyo.Constraint(model.h, rule=delay_nuc_power_constraint_rule)

#############
### HYDRO ###
#############

def lake_disagregate_gene_rule(model, h):
    """Explicit relation between global 'lake' tec and individual simplified regional dams & reservoirs"""
    # Note : model.capa['lake'] is fixed by hypothesis and equal to the sum of input capa of dams & lakes, hence a constraint is not required

    return model.gene['lake', h] == sum(model.lake_gene[lake, h] for lake in model.lakes)
model.lake_disagregate_gene = pyo.Constraint(model.h, rule=lake_disagregate_gene_rule)

def lake_generation_rule(model, h, lake):
    """Get constraint on maximum power for each individual simplified regional dam & reservoir"""

    return model.lake_gene[lake, h] <= lake_max_capa[lake, h]
model.lake_generation = pyo.Constraint(model.h, model.lakes, rule=lake_generation_rule)

def lake_stored_capacity_constraint_rule(model, h, lake):
    """Get constraint on maximum energy that is stored in each individual simplified regional dam & reservoir"""

    return model.lake_stored[lake, h] <= lake_capacity[lake]
model.lake_stored_capacity_constraint = pyo.Constraint(model.h, model.lakes, rule=lake_stored_capacity_constraint_rule)

def lake_minimal_stored_capacity_constraint_rule(model, h, lake):
    """Set a contraint on minimal energy that must be stored for recreative purposes during summer"""

    return model.lake_stored[lake, h] >= lake_minimal_capacity[lake, h]
model.lake_minimal_stored_capacity_constraint = pyo.Constraint(model.h, model.lakes, rule=lake_minimal_stored_capacity_constraint_rule)

def lake_storing_constraint_rule(model, h, lake):
    """Set the evolution of dams & reservoirs stocks"""

    hPOne = h+1 if h < model.h.last() else 0
    charge = lake_inflows[lake, h]
    discharge = model.lake_gene[lake, h] + model.lake_spilled[lake, h] - lake_prod_to_be_spilled[lake, h]
    return model.lake_stored[lake, hPOne] == model.lake_stored[lake, h] + charge - discharge
model.lake_storing_constraint = pyo.Constraint(model.h, model.lakes, rule=lake_storing_constraint_rule)

def lake_spillage_constraint_rule(model, h, lake):
    """Defines spilled energy"""

    return model.lake_spilled[lake, h] >= lake_prod_to_be_spilled[lake, h] - model.lake_gene[lake, h]
model.lake_spillage_constraint = pyo.Constraint(model.h, model.lakes, rule=lake_spillage_constraint_rule)

"""SOLVE STATEMENT

Choice of the solver.
You can remove the '#' in the forth line to display the output of the solver.
"""

opt = SolverFactory('gurobi')
results = opt.solve(model, options={'Presolve':1, 'LogFile':"grblogfile_eoles_"+model_name})
# For detailed solver log file, add the following option : 'LogFile':"grblogfile_eoles_"+model_name
# For use with Inari, add : 'Threads':16 to limit the load
# model.display()
print("\nObjective function value : "+str(pyo.value(model.objective))+"\n")

"""SET OUTPUTS VARIABLES"""

    # Overall yearly energy generated by the technology in TWh
gene_per_tec = {}
for tec in model.tec:
    gene_per_tec[tec] = sum(pyo.value(model.gene[tec,h]) for h in model.h) / 1000 / nb_years

    # The whole yearly demand in TWh
sumdemand = sum(demand[h] for h in model.h) / 1000 / nb_years
    # The whole generation in TWh
sumgene = sum(gene_per_tec[gen] for gen in model.gen)

    # The yearly input for storage in TWh
nSTORAGE = {}
for storage in model.str:
    nSTORAGE[storage] = sum(pyo.value(model.storage[storage,h]) for h in model.h) / 1000 /nb_years

    # Yearly electricity cost per MWh produced (euros/MWh)
lcoe_sys1 = pyo.value(model.objective) /nb_years * 1000 / sumgene

    # Yearly storage related loss in % of power production and in TWh
str_loss_TWh = gene_per_tec['electrolysis']/eta_out['electrolysis'] - miscellaneous['H2_demand']/1000/eta_out['electrolysis'] - gene_per_tec['h2_ccgt']
str_loss_TWh += gene_per_tec['methanation']*(1/eta_out['methanation'] - 1)
for storage in model.str:
    if storage != 'hydrogen':
        str_loss_TWh += nSTORAGE[storage] - gene_per_tec[storage]
str_loss_percent = 100 * str_loss_TWh / sumgene

    # Yearly load curtailment in % of power production and in TWh
lc_TWh = 0
for h in model.h:
    gene_electrolysis = pyo.value(model.gene['electrolysis',h]) / eta_out['electrolysis']
    gene_methanation = pyo.value(model.gene['methanation',h]) / eta_out['methanation']
    storage = sum(pyo.value(model.storage[str, h]) for str in model.str if (str != "hydrogen" and str != "methane"))
    balance = sum(pyo.value(model.gene[gen, h]) for gen in model.gen) - (demand[h] + gene_electrolysis + gene_methanation + storage)
    if balance > 0:
        lc_TWh += balance / 1000
lc_TWh = lc_TWh/nb_years
lc_percent = 100 * lc_TWh / sumgene

    # Dual values
spot_price = {}
for h in model.h:
    spot_price[h] = - 1000000 * model.dual[model.adequacy_constraint[h]]

    # Average cost of hydrogen (euros/kg)
lcoh_1 = (pyo.value(model.capa['electrolysis']) - existing_capa['electrolysis']) * annuities['electrolysis']
lcoh_2 = pyo.value(model.capa['electrolysis']) * fOM['electrolysis']
lcoh_3 =  sum(pyo.value(model.gene['electrolysis',h])*(vOM['electrolysis']+\
    spot_price[h]/1000/eta_out['electrolysis']) for h in model.h)/nb_years
lcoh_4 = storage_annuities['hydrogen'] * (pyo.value(model.capacity['hydrogen']) - existing_capacity['hydrogen'])
lcoh = 0 if gene_per_tec['electrolysis']==0 else (lcoh_1 + lcoh_2 + lcoh_3 + lcoh_4) * miscellaneous["pci_h2"] / gene_per_tec['electrolysis'] /1000

    # Yearly average cost of methane (euros/kg)
lcom_1 = 0 if gene_per_tec['pyrogazification']==0 else pyo.value(model.capa['pyrogazification'])*(annuities['pyrogazification']+fOM['pyrogazification'])/gene_per_tec['pyrogazification'] \
    + vOM['pyrogazification']/nb_years
lcom_2 = 0 if gene_per_tec['methanization']==0 else pyo.value(model.capa['methanization'])*(annuities['methanization']+fOM['methanization'])/gene_per_tec['methanization'] \
    + vOM['methanization']/nb_years
lcom_3 = 0 if gene_per_tec['methanation']==0 else pyo.value(model.capa['methanation'])*(annuities['methanation']+fOM['methanation'])/gene_per_tec['methanation'] \
    + vOM['methanation']/nb_years + sum(pyo.value(model.gene['methanation',h])*spot_price[h]/1000/eta_out['methanation'] for h in model.h)/nb_years
lcom_4 = 0 if gene_per_tec['pyrogazification']+gene_per_tec['methanization']+gene_per_tec['methanation'] == 0 else \
    storage_annuities['methane']*pyo.value(model.capacity['methane']) / (gene_per_tec['pyrogazification'] + gene_per_tec['methanization'] + gene_per_tec['methanation'])
lcom = (lcom_1 + lcom_2 + lcom_3 + lcom_4) * miscellaneous['pci_ch4'] / 1000

    # Yearly electricity cost per MWh consumed (euros/MWh)
loss_h2 = lcoh * miscellaneous['H2_demand'] / miscellaneous["pci_h2"] / 1000
loss_ch4 = lcom * miscellaneous['CH4_demand'] / miscellaneous["pci_ch4"] / 1000
lcoe_sys2 = (pyo.value(model.objective)*1000/nb_years - loss_h2 - loss_ch4) / sumdemand

"""OUTPUTS
    There are 4 output files :
        - Summary           : A little summary with the cost and some others data
        - Hourly-Generation : Hourly data
        - Elec_Balance      : Electric Production and Consumption
        - Capacities        : List of capacities by technologies

The try, except loop is here the manage error in the creation of the outputs.
"""

#Summary
summary_file = "outputs/eoles_" + model_name + "_summary.csv"
try:
    with open(summary_file,"w",newline="") as summary:
        summary_writer = csv.writer(summary)

        summary_header = ["COST","LCOH","LCOM","LCOE_SYS1","LCOE_SYS2","STR_LOSS","","LC",""]
        summary_writer.writerow(summary_header)

        summary_definition = [
            "in billion euro",
            "average cost of hydrogen (euros/kg)",
            "average cost of methane (euros/kg)",
            "electricity cost per MWh produced (euros/MWh)",
            "electricity cost per MWh consumed (euros/MWh)",
            "yearly storage related loss in % of power production",
            "yearly storage related loss in TWh",
            "load curtailment in % of power production",
            "load curtailment in TWh"]
        summary_writer.writerow(summary_definition)

        summary_data = [
            round(pyo.value(model.objective)/nb_years,4),
            round(lcoh,2) if lcoh != 0 else '-',
            round(lcom,2) if lcom != 0 else '-',
            round(lcoe_sys1,2),
            round(lcoe_sys2,2),
            round(str_loss_percent,2),
            round(str_loss_TWh,2),
            round(lc_percent,2),
            round(lc_TWh,2)]
        summary_writer.writerow(summary_data)
except Exception as e:
    if e.args == (13, 'Permission denied') :
        sys.stderr.write("Error : Permission Denied")
        print("Error : Permission Denied. Maybe try to close the file : "+summary_file) 
    else : 
        print("There is an Error (summary) : "+e.args[0])

#Hourly_Generation
hourly_file = "outputs/eoles_" + model_name + "_hourly_generation.csv"
try:
    with open(hourly_file,"w",newline="") as hourly:
        hourly_writer = csv.writer(hourly)
        ###
        hourly_title = ["","Generation ->"]
        for i in range(len(model.tec)-1):
            hourly_title.append("")
        for i in range(len(model.lakes)):
            hourly_title.append("")
        hourly_title += [
            "Consumption ->",
            "Storage input ->","","","","",
            "State of charge ->","","","",""]
        for i in range(len(model.lakes)):
            hourly_title += [""]
        hourly_title +=["","","Dual Value", "Reserves ->"]
        hourly_writer.writerow(hourly_title)
        ###
        hourly_header = ["hour"]
        for tec in model.tec:
            if tec != "lake":
                hourly_header.append(tec)
        hourly_header += ["lake","including :"]
        for i in range(len(model.lakes)-1):
            hourly_header += [""]
        hourly_header += ["electricity_demand"]
        for tec in model.str:
            hourly_header += [tec+'_input']
        for tec in model.str:
            hourly_header += [tec+'_stored']
        for lake in model.lakes:
            hourly_header += [lake+"_stored"]
        hourly_header += [
            "nuc_on",
            "nuc_cf",
            "elec balance"]
        for rsv in model.frr:
            hourly_header.append(rsv)
        hourly_writer.writerow(hourly_header)
        ###
        hourly_definition = [""]
        for tec in model.tec:
            if tec != "lake":
                hourly_definition.append(technologies_definition.get(tec))
        hourly_definition.append(technologies_definition.get("lake"))
        for lake in model.lakes:
            hourly_definition.append(lake)
        hourly_definition += ["electricity_demand"]
        for storage_tec in model.str:
            hourly_definition.append(technologies_definition.get(storage_tec))
        for storage_tec in model.str:
            hourly_definition.append(technologies_definition.get(storage_tec))
        for lake in model.lakes:
            hourly_definition += [lake]
        hourly_definition += [
            "Available nuclear power",
            "NPP fleet-wide capacity factor",
            "/"]
        for rsv in model.frr:
            hourly_definition.append(technologies_definition.get(rsv))
        hourly_writer.writerow(hourly_definition)
        ###
        hourly_units = [""]
        for tec in model.tec:
            if tec != "lake":
                hourly_units.append(technologies_units.get(tec))
        hourly_units += ["GWh-e"]
        for i in range(len(model.lakes)):
            hourly_units += ["GWh-e"]
        hourly_units += ["GWh-e"]
        for storage_tec in model.str:
            hourly_units.append(technologies_units.get(storage_tec))
        for storage_tec in model.str:
            hourly_units.append(technologies_units.get(storage_tec))
        for lake in model.lakes:
            hourly_units += ["GWh-e"]
        hourly_units += [
            "%",
            "%",
            "/"]
        for rsv in model.frr:
            hourly_units.append(technologies_units.get(rsv))
        hourly_writer.writerow(hourly_units)
        ###
        for h in model.h:
            hourly_data = [h]
            for tec in model.tec:
                if tec != "lake":
                    hourly_data.append(round(pyo.value(model.gene[tec,h]),2))
            
            hourly_data.append(round(pyo.value(model.gene["lake",h]),2))
            for lake in model.lakes:
                hourly_data.append(round(pyo.value(model.lake_gene[lake,h]),2))

            hourly_data.append(round(demand[h],2))
            
            for storage_tecs in model.str:
                hourly_data.append(round(pyo.value(model.storage[storage_tecs,h]),2))
            
            for storage_tecs in model.str:
                hourly_data.append(round(pyo.value(model.stored[storage_tecs,h]),2))
            
            for lake in model.lakes:
                hourly_data.append(round(pyo.value(model.lake_stored[lake,h]),2))

            nuc_on = nuc_max_operating_capa[h] - pyo.value(model.capa_nuc_off[h])
            #hourly_data.append(round(pyo.value(model.capa_nuc_off[hour]), 2)) 
            if pyo.value(model.capa['nuc']) == 0: hourly_data.append(0)
            else: hourly_data.append(round(100*nuc_on/pyo.value(model.capa['nuc']), 2))
            
            #hourly_data.append(round(pyo.value(model.helper_positive[hour]), 2))
            if nuc_on == 0: hourly_data.append(0)
            else: hourly_data.append(round(100*pyo.value(model.gene['nuc', h])/nuc_on, 2))
            
            hourly_data.append(round(spot_price[h],2))
            
            for frr in model.frr:
                hourly_data.append(round(pyo.value(model.reserve[frr,h]),2))

            hourly_writer.writerow(hourly_data)
except Exception as e:
    if e.args == (13, 'Permission denied') :
        sys.stderr.write("Error : Permission Denied")
        print("Error : Permission Denied. Maybe try to close the file : "+hourly_file) 
    else : 
        print("There is an Error (hourly) : "+e.args[0])

#Elec_balance
elec_balance_file = "outputs/eoles_" + model_name + "_elec_balance.csv"
try:
    with open(elec_balance_file,"w",newline="") as balance:
        balance_writer = csv.writer(balance)
        balance_title = ["Supply in TWh-e ->"]
        balance_writer.writerow(balance_title)
        balance_header = []
        for tec in model.tec:
            if (tec not in model.biogas) and (tec not in model.hydrogen):
                balance_header.append(tec)
        balance_writer.writerow(balance_header)
        balance_definition = []
        for tec in model.tec:
            if (tec not in model.biogas) and (tec not in model.hydrogen):
                balance_definition.append(technologies_definition.get(tec))
        balance_writer.writerow(balance_definition)
        balance_data = []
        for tec in model.tec:
            if (tec not in model.biogas) and (tec not in model.hydrogen):
                balance_data.append(round(pyo.value(gene_per_tec[tec]),2))
        balance_writer.writerow(balance_data)
        ################################################
        balance_writer.writerow([])
        balance_title = ["Use in TWh-e ->"]
        balance_writer.writerow(balance_title)
        balance_header = ['electrolysis', 'methanation']
        for tec in model.str:
            if tec != 'hydrogen' and tec != 'methane':
                balance_header.append(tec)
        balance_header += ["elec demand","load curtailment"]
        balance_writer.writerow(balance_header)
        balance_data = [round(pyo.value(gene_per_tec['electrolysis'])/eta_out['electrolysis'],2), round(pyo.value(gene_per_tec['methanation'])/eta_out['methanation'],2)]
        for tec in model.str:
            if tec != 'hydrogen' and tec != 'methane':
                balance_data.append(round(nSTORAGE[tec],2))
        balance_data += [round(sumdemand,2),round(lc_TWh,2)]
        balance_writer.writerow(balance_data)
        ################################################
        balance_writer.writerow([])
        balance_title = ["Hours in use ->"]
        balance_writer.writerow(balance_title)
        balance_header = ['electrolysis', 'methanation']
        for tec in model.str:
            if tec != 'hydrogen' and tec != 'methane':
                balance_header.append(tec)
        balance_writer.writerow(balance_header)
        balance_data = []
        for tec in balance_header:
            hours_in_use = 0
            for h in model.h:
                if pyo.value(model.gene[tec,h]) > 0 :
                    hours_in_use += 1
            balance_data.append(round(hours_in_use/nb_years))
        balance_writer.writerow(balance_data)
        #####################################################
        balance_writer.writerow([])
        balance_title = ["Capacity Factor in % ->"]
        balance_writer.writerow(balance_title)
        balance_tec = ['lake','ocgt','ccgt','nuc','h2_ccgt','electrolysis','battery1', 'battery4','phs']
        balance_writer.writerow(balance_tec)
        balance_data = []
        for tec in balance_tec:
            if pyo.value(model.capa[tec]) == 0:
                capa_factor = "-"
            else:
                capa_factor = round(gene_per_tec[tec]/pyo.value(model.capa[tec])*1000/8760*100,2)
            balance_data.append(capa_factor)
        balance_writer.writerow(balance_data)
        ##################################################
except Exception as e:
    if e.args == (13, 'Permission denied') :
        sys.stderr.write("Error : Permission Denied")
        print("Error : Permission Denied. Maybe try to close the file : "+elec_balance_file) 
    else : 
        print("There is an Error (balance) : "+e.args[0])

#Biogas balance
#TO DO

#Hydrogen balance
#TO DO

#Capacities
capacities_file = "outputs/eoles_" + model_name + "_capacities.csv"
try:
    with open(capacities_file,"w",newline="") as capacities:
        capacities_writer = csv.writer(capacities)
        ################################################################
        capacities_title = ["Capacity in GW ->"]
        capacities_writer.writerow(capacities_title)
        capacities_header = []
        for tec in model.tec:
            capacities_header.append(tec)
        capacities_writer.writerow(capacities_header)
        capacities_definition = []
        for tec in model.tec:
            capacities_definition.append(technologies_definition.get(tec))
        capacities_writer.writerow(capacities_definition)
        capacities_data = []
        for tec in model.tec:
            capacities_data.append(round(pyo.value(model.capa[tec]),2))
        capacities_writer.writerow(capacities_data)
        ################################################################
        capacities_writer.writerow([])
        capacities_title = ["Energy volume of storage technology in GWh ->"]
        capacities_writer.writerow(capacities_title)
        capacities_header = []
        for storages_tecs in model.str:
            capacities_header.append(storages_tecs)
        capacities_writer.writerow(capacities_header)
        capacities_definition = []
        for str_def in model.str:
            capacities_definition.append(technologies_definition.get(str_def))
        capacities_writer.writerow(capacities_definition)
        capacities_data = []
        for storages_tecs in model.str:
            capacities_data.append(round(pyo.value(model.capacity[storages_tecs]),2))
        capacities_writer.writerow(capacities_data)
except Exception as e:
    if e.args == (13, 'Permission denied') :
        sys.stderr.write("Error : Permission Denied")
        print("Error : Permission Denied. Maybe try to close the file : "+capacities_file) 
    else : 
        print("There is an Error (capacities) : "+e.args[0])

print("Execution time (EOLES) : " + str(round(time.time() - start_time, 2)) + "s")
