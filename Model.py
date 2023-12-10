import pandas as pd
import numpy as np
import logging
import json
import os
import math
from utils import get_pandas, process_RTE_demand, calculate_annuities_capex, calculate_annuities_storage_capex, \
    update_ngas_cost, define_month_hours, calculate_annuities_renovation, get_technical_cost, extract_hourly_generation, \
    extract_spot_price, extract_capacities, extract_energy_capacity, extract_supply_elec, extract_primary_gene, \
    extract_use_elec, extract_renovation_rates, extract_heat_gene,   calculate_LCOE_gene_tec, calculate_LCOE_conv_tec, \
    extract_charging_capacity, extract_annualized_costs_investment_new_capa, extract_CH4_to_power, extract_power_to_CH4, \
    extract_power_to_H2, extract_peak_load, extract_peak_heat_load, extract_annualized_costs_investment_new_capa_nofOM, \
    extract_functionment_cost, extract_carbon_value, extract_H2_to_power, get_carbon_content
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Set,
    NonNegativeReals,  # a verifier, mais je ne pense pas que ce soit une erreur
    Constraint,
    SolverFactory,
    Suffix,
    Var,
    Objective,
    value
)



class ModelEOLES():
    def __init__(self, name, config, path, logger, 
                 hourly_heat_elec, hourly_heat_gas, 
                 hourly_heat_district=None,
                 wood_consumption=0, oil_consumption=0,
                 existing_capacity=None, 
                 existing_charging_capacity=None, 
                 existing_energy_capacity=None, 
                 maximum_capacity=None,
                 method_hourly_profile="valentin", 
                 anticipated_social_cost_of_carbon=0, 
                 actual_social_cost_of_carbon=0, 
                 year=2050, anticipated_year=2050,
                 scenario_cost=None, 
                 existing_annualized_costs_elec=0,
                 existing_annualized_costs_CH4=0, 
                 existing_annualized_costs_H2=0, 
                 existing_annualized_costs_CH4_naturalgas=0,
                 existing_annualized_costs_CH4_biogas=0, 
                 carbon_constraint=False, discount_rate=0.045, 
                 calibration=False):

        self.name = name
        self.config = config
        self.logger = logger
        self.path = path
        self.model = ConcreteModel()
        # Dual Variable, used to get the marginal value of an equation.
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.nb_years = self.config["nb_years"]
        self.input_years = self.config["input_years"]
        self.anticipated_scc = anticipated_social_cost_of_carbon
        self.actual_scc = actual_social_cost_of_carbon
        self.discount_rate = discount_rate
        self.year = year
        self.carbon_constraint = carbon_constraint
        self.capacity_factor_nuclear = self.config["capacity_factor_nuclear"]
        self.capacity_factor_nuclear_hourly = self.config['capacity_factor_nuclear_hourly']  # capacity factor which applies for all hours and not only total
        self.hourly_ramping_nuclear = self.config['hourly_ramping_nuclear']
        self.anticipated_year = anticipated_year
        self.existing_annualized_costs_elec = existing_annualized_costs_elec
        self.existing_annualized_costs_CH4 = existing_annualized_costs_CH4
        self.existing_annualized_costs_CH4_naturalgas = existing_annualized_costs_CH4_naturalgas
        self.existing_annualized_costs_CH4_biogas = existing_annualized_costs_CH4_biogas
        self.existing_annualized_costs_H2 = existing_annualized_costs_H2
        self.calibration = calibration  # whether we rely on the coupling for the forecast of electricity demand or not

        assert hourly_heat_elec is not None, "Hourly electricity heat profile missing."
        assert hourly_heat_gas is not None, "Hourly gas heat profile missing."

        # loading exogeneous variable data
        if self.calibration:
            data_hourly_and_anticipated = read_hourly_data(config, self.anticipated_year, method=method_hourly_profile, calibration=self.calibration, hourly_heat_elec=hourly_heat_elec)
        else:  # classical setting
            data_hourly_and_anticipated = read_hourly_data(config, self.anticipated_year, method=method_hourly_profile, calibration=self.calibration)
        self.load_factors = data_hourly_and_anticipated["load_factors"]
        self.elec_demand1y = data_hourly_and_anticipated["demand"]
        self.lake_inflows = data_hourly_and_anticipated["lake_inflows"]
        assert int(self.load_factors.shape[0]/(8760*6)) == self.nb_years, "Specified number of years does not match load factors"

        self.hourly_heat_elec = hourly_heat_elec

        if self.nb_years == 1:
            self.elec_demand1y = self.elec_demand1y + self.hourly_heat_elec  # we add electricity demand from residential heating
            self.elec_demand = self.elec_demand1y
            for i in range(self.nb_years - 1):  # plus nécessaire avec la condition if /else
                self.elec_demand = pd.concat([self.elec_demand, self.elec_demand1y], ignore_index=True)
        else:  # nb_years > 1
            self.elec_demand = pd.Series()
            for i in range(self.nb_years):
                self.elec_demand_year = self.elec_demand1y + self.hourly_heat_elec.loc[0:8760]  # TODO: a changer selon format Lucas pour ajouter la thermosensibilité
                self.elec_demand = pd.concat([self.elec_demand, self.elec_demand_year], ignore_index=True)

        # Creation of demand for district heating
        if hourly_heat_district is not None:
            self.district_heating_demand = hourly_heat_district
            for i in range(self.nb_years - 1):
                self.district_heating_demand = pd.concat([self.district_heating_demand, hourly_heat_district], ignore_index=True)
        else:  # if not specified, this is equal to 0
            self.district_heating_demand = pd.Series(0, index=self.elec_demand.index)
        self.hourly_heat_gas = hourly_heat_gas
        self.wood_consumption = wood_consumption
        self.oil_consumption = oil_consumption

        self.H2_demand = {}
        self.CH4_demand = {}

        if self.hourly_heat_gas is not None:  # we provide hourly gas data
            # TODO: a changer selon le format choisi par Lucas pour nb_years > 1, pour ajouter la thermosensibilité
            self.gas_demand = self.hourly_heat_gas
            for i in range(self.nb_years - 1):
                self.gas_demand = pd.concat([self.gas_demand, self.hourly_heat_gas], ignore_index=True)

        # loading exogeneous static data
        # data_static = read_input_static(self.config, self.year)
        data_technology = read_technology_data(self.config, self.year)  # get current technology data
        data_annual = read_annual_data(self.config, self.anticipated_year)  # get anticipated demand and energy prices
        data_technology.update(data_annual)
        data_static = data_technology
        if scenario_cost is not None:  # we update costs based on data given in scenario
            for df in scenario_cost.keys():
                if df == "existing_capacity" and existing_capacity is not None:
                    for tec in scenario_cost[df].keys():
                        existing_capacity[tec] = scenario_cost[df][tec]
                if df == "existing_charging_capacity" and existing_charging_capacity is not None:
                    for tec in scenario_cost[df].keys():
                        existing_charging_capacity[tec] = scenario_cost[df][tec]
                if df == "existing_energy_capacity" and existing_energy_capacity is not None:
                    for tec in scenario_cost[df].keys():
                        existing_energy_capacity[tec] = scenario_cost[df][tec]
                if df == "maximum_capacity" and maximum_capacity is not None:
                    for tec in scenario_cost[df].keys():
                        maximum_capacity[tec] = scenario_cost[df][tec]
                for tec in scenario_cost[df].keys():
                    data_static[df][tec] = scenario_cost[df][tec]

        self.epsilon = data_static["epsilon"]
        if existing_capacity is not None:
            self.existing_capacity = existing_capacity
        else:  # default value
            self.existing_capacity = data_static["existing_capacity"]
        if existing_charging_capacity is not None:
            self.existing_charging_capacity = existing_charging_capacity
        else:  # default value
            self.existing_charging_capacity = data_static["existing_charging_capacity"]
        if existing_energy_capacity is not None:
            self.existing_energy_capacity = existing_energy_capacity
        else:  # default value
            self.existing_energy_capacity = data_static["existing_energy_capacity"]
        if maximum_capacity is not None:
            self.maximum_capacity = maximum_capacity
        else:
            self.maximum_capacity = data_static["maximum_capacity"]
        self.maximum_charging_capacity = data_static["maximum_charging_capacity"]
        self.maximum_energy_capacity = data_static["maximum_energy_capacity"]
        self.fix_capacities = data_static["fix_capacities"]
        self.fix_charging_capacities = data_static["fix_charging_capacities"]
        self.fix_energy_capacities = data_static["fix_energy_capacities"]
        self.lifetime = data_static["lifetime"]
        self.construction_time = data_static["construction_time"]
        self.capex = data_static["capex"]
        self.storage_capex = data_static["storage_capex"]
        self.fOM = data_static["fOM"]
        self.vOM = data_static["vOM"]
        self.eta_in = data_static["eta_in"]
        self.eta_out = data_static["eta_out"]
        self.conversion_efficiency = data_static["conversion_efficiency"]
        self.miscellaneous = data_static["miscellaneous"]
        self.prediction_transport_and_distrib_annuity = data_static["prediction_transport_and_distrib_annuity"]
        self.prediction_transport_offshore_annuity = data_static["prediction_transport_offshore_annuity"]
        self.biomass_potential = data_static["biomass_potential"]
        self.district_heating_potential = data_static["district_heating_potential"]
        self.total_H2_demand = data_static["demand_H2_RTE"]
        self.energy_prices = data_static["energy_prices"]
        self.carbon_budget = data_static["carbon_budget"]
        self.vOM["wood"], self.vOM["oil"] = self.energy_prices["wood"] * 1e-3, self.energy_prices["oil"] * 1e-3  # €/kWh
        self.vOM["natural_gas"], self.vOM['coal'] = self.energy_prices["natural_gas"] * 1e-3, self.energy_prices["coal"] * 1e-3

        # calculate annuities
        self.annuities = calculate_annuities_capex(self.discount_rate, self.capex, self.construction_time,
                                                   self.lifetime)
        self.storage_annuities = calculate_annuities_storage_capex(self.discount_rate, self.storage_capex,
                                                                   self.construction_time, self.lifetime)

        if not self.carbon_constraint:  # on prend en compte le scc mais pas de contrainte sur le budget
            # Update natural gaz vOM based on social cost of carbon
            self.vOM.loc["natural_gas"] = update_ngas_cost(self.vOM.loc["natural_gas"], scc=self.anticipated_scc, emission_rate=0.2295)  # €/kWh
            self.vOM["oil"] = update_ngas_cost(self.vOM["oil"], scc=self.anticipated_scc, emission_rate=0.324)  # to check !!
            self.vOM["coal"] = update_ngas_cost(self.vOM["coal"], scc=self.anticipated_scc, emission_rate=0.986)
            self.vOM["wood"] = update_ngas_cost(self.vOM["wood"], scc=self.anticipated_scc, emission_rate=0)  # to check !!

        # defining needed time steps
        self.first_hour = 0
        self.last_hour = len(self.elec_demand)
        self.first_month = self.miscellaneous['first_month']

        self.hours_by_months = {1: 744, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 720, 10: 744,
                                11: 720, 12: 744}

        self.months_hours = {1: range(0, self.hours_by_months[self.first_month])}
        self.month_hours = define_month_hours(self.first_month, self.nb_years, self.months_hours, self.hours_by_months)
        # for i in range(1, self.nb_years):  # we update month_hours to add multiple years
        #     new_month_hours = {key + 12*i: self.months_hours[key] for key in self.months_hours.keys()}
        #     self.months_hours.update(new_month_hours)


    def define_sets(self):
        # Range of hour
        self.model.h = \
            RangeSet(self.first_hour, self.last_hour - 1)
        # Months
        self.model.months = \
            RangeSet(1, 12 * self.nb_years)
        # Years
        self.model.years = RangeSet(0, self.nb_years - 1)

        # Technologies
        self.model.tec = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "methanization",
                            "ocgt", "ccgt", "nuclear", "h2_ccgt", "phs", "battery1", "battery4",
                            "methanation", "pyrogazification", "electrolysis", "natural_gas", "coal", "hydrogen", "methane",
                            "geothermal", "central_gas_boiler", "central_wood_boiler", "uiom", "CTES"])
        # Variables Technologies
        self.model.vre = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river"])
        # Electricity generating technologies
        self.model.elec_balance = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "phs",
                            "battery1", "battery4", "ocgt", "ccgt", "h2_ccgt", "coal"])

        # Technologies for upward FRR
        self.model.frr = Set(initialize=["lake", "phs", "ocgt", "ccgt", "nuclear", "h2_ccgt"])

        # Technologies producing electricity (not including storage technologies)
        # self.model.elec_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
        #                                        "nuclear", "ocgt", "ccgt", "h2_ccgt"])
        self.model.elec_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
                                               "nuclear"])
        self.model.CH4_gene = Set(initialize=["methanization", "pyrogazification", "methanation", "natural_gas"])
        self.model.H2_gene = Set(initialize=["electrolysis"])
        # Primary energy production
        self.model.primary_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river",
                                                  "lake", "nuclear", "methanization", "pyrogazification",
                                                  "natural_gas"])
        # Technologies using electricity
        self.model.use_elec = Set(initialize=["phs", "battery1", "battery4", "electrolysis"])
        # Technologies producing gas
        self.model.gas_gene = Set(initialize=["methanization", "pyrogazification"])

        # Gas technologies used for balance (both CH4 and H2)
        self.model.CH4_balance = Set(
            initialize=["methanization", "pyrogazification", "natural_gas", "methanation", "methane"])
        self.model.CH4_balance_historic = Set(initialize=["natural_gas", "methane"])
        self.model.CH4_balance_biogas = Set(initialize=["methanization", "pyrogazification", "methanation"])
        self.model.H2_balance = Set(initialize=["electrolysis", "hydrogen"])

        # District heating technologies
        # TODO: voir si on ajoute les UIOM dans la modélisation !!
        self.model.district_heating_balance = Set(initialize=["geothermal", "central_gas_boiler", "central_wood_boiler", "CTES"])

        # Conversion technologies
        self.model.from_elec_to_CH4 = Set(initialize=["methanation"])
        self.model.from_elec_to_H2 = Set(initialize=["electrolysis"])
        self.model.from_CH4_to_elec = Set(initialize=["ocgt", "ccgt"])
        self.model.from_H2_to_elec = Set(initialize=["h2_ccgt"])

        # Storage technologies
        self.model.str = \
            Set(initialize=["phs", "battery1", "battery4", "hydrogen", "methane", "CTES"])
        # Electricity storage Technologies
        self.model.str_elec = Set(initialize=["phs", "battery1", "battery4"])
        # Battery Storage
        self.model.battery = Set(initialize=["battery1", "battery4"])
        # CH4 storage
        self.model.str_CH4 = Set(initialize=["methane"])
        # H2 storage
        self.model.str_H2 = Set(initialize=["hydrogen"])
        # Central heat storage
        self.model.str_district_heating = Set(initialize=["CTES"])

    def define_other_demand(self):
        # Set the hydrogen demand for each hour
        for hour in self.model.h:
            # self.H2_demand[hour] = self.miscellaneous['H2_demand']
            self.H2_demand[hour] = self.total_H2_demand / 8760  # We make the assumption that H2 demand profile is flat, and is given with an annual value

        # Set the methane demand for each hour
        for hour in self.model.h:
            self.CH4_demand[hour] = self.gas_demand[hour]  # a bit redundant, could be removed

    def define_variables(self):

        def capacity_bounds(model, i):
            if i in self.maximum_capacity.keys():  # there exists a max capacity
                return self.existing_capacity[i], self.maximum_capacity[
                    i]  # existing capacity is always the lower bound
            else:
                return self.existing_capacity[i], None  # in this case, only lower bound exists

        def charging_capacity_bounds(model, i):
            # TODO: j'ai enlevé cette contrainte, car je suppose ici que la seule contrainte provient de la discharging capacity
            # if i in self.maximum_charging_capacity.keys():
            #     return self.existing_charging_capacity[i], self.maximum_capacity[i]
            # else:
            return self.existing_charging_capacity[i], None

        def energy_capacity_bounds(model, i):
            if i in self.maximum_energy_capacity.keys():
                return self.existing_energy_capacity[i], self.maximum_energy_capacity[i]
            else:
                return self.existing_energy_capacity[i], None

        # Hourly energy generation in GW

        self.model.gene = \
            Var(((tec, h) for tec in self.model.tec for h in self.model.h), within=NonNegativeReals, initialize=0)

        # Overall yearly installed capacity in GW
        self.model.capacity = \
            Var(self.model.tec, within=NonNegativeReals, bounds=capacity_bounds)

        # Charging power capacity of each storage technology in GW
        self.model.charging_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=charging_capacity_bounds)

        # Energy volume of storage technology in GWh
        self.model.energy_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=energy_capacity_bounds)

        # Hourly electricity input of battery storage GWh
        self.model.storage = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Energy stored in each storage technology in GWh = Stage of charge
        self.model.stored = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Required upward frequency restoration reserve in GW
        self.model.reserve = \
            Var(((reserve, h) for reserve in self.model.frr for h in self.model.h), within=NonNegativeReals,
                initialize=0)

    def fix_values(self):
        for tec in self.model.tec:
            if tec in self.fix_capacities.keys():
                self.model.capacity[tec].fix(self.fix_capacities[tec])
        for tec in self.model.tec:
            if tec in self.fix_charging_capacities.keys():
                self.model.charging_capacity[tec].fix(self.fix_charging_capacities[tec])
        for tec in self.model.tec:
            if tec in self.fix_energy_capacities.keys():
                self.model.energy_capacity[tec].fix(self.fix_energy_capacities[tec])

    def define_constraints(self):
        def generation_vre_constraint_rule(model, h, vre):
            """Cnstraint on variables renewable profiles generation."""
            return model.gene[vre, h] == model.capacity[vre] * self.load_factors[vre, h]

        def generation_nuclear_constraint_rule(model, y):
            """Constraint on total nuclear production which cannot be superior to nuclear capacity times a given
            capacity factor inferior to 1."""
            return sum(model.gene["nuclear", h] for h in range(8760*y,8760*(y+1)-1)) <= self.capacity_factor_nuclear * model.capacity["nuclear"] * 8760

        def generation_nuclear_constraint_hourly_rule(model, h):
            """Constraint on nuclear production which cannot be superior to nuclear capacity times a given capacity factor.
            This holds for all hours."""
            return model.capacity['nuclear'] * self.capacity_factor_nuclear_hourly >= model.gene['nuclear', h]

        def generation_capacity_constraint_rule(model, h, tec):
            """Constraint on maximum power for non-VRE technologies."""
            return model.capacity[tec] >= model.gene[tec, h]

        def battery1_capacity_constraint_rule(model):
            """Constraint on capacity of battery 1h."""
            # TODO: check that the constraint is ok
            return model.capacity['battery1'] == model.energy_capacity['battery1']

        def battery4_capacity_constraint_rule(model):
            """Constraint on capacity of battery 4h."""
            # TODO: check that the constraint is ok
            return model.capacity['battery4'] == model.energy_capacity['battery4'] / 4

        def frr_capacity_constraint_rule(model, h, frr):
            """Constraint on maximum generation including reserves"""
            return model.capacity[frr] >= model.gene[frr, h] + model.reserve[frr, h]

        def storing_constraint_rule(model, h, storage_tecs):
            """Constraint on energy storage consistency."""
            hPOne = h + 1 if h < (self.last_hour - 1) else 0
            charge = model.storage[storage_tecs, h] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, h] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return model.stored[storage_tecs, hPOne] == model.stored[storage_tecs, h] + flux

        def storage_first_last_constraint_rule(model, storage_tecs):
            """Constraint on stored energy to be equal at the end and at the start."""
            first = model.stored[storage_tecs, self.first_hour]
            last = model.stored[storage_tecs, self.last_hour - 1]
            charge = model.storage[storage_tecs, self.last_hour - 1] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, self.last_hour - 1] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return first == last + flux

        def lake_reserve_constraint_rule(model, month):
            """Constraint on maximum monthly lake generation. Total generation from lake over a month cannot exceed
            a certain given value."""
            return sum(model.gene['lake', hour] for hour in self.months_hours[month]) <= self.lake_inflows[month] * 1000

        def stored_capacity_constraint(model, h, storage_tecs):
            """Constraint on maximum energy that is stored in storage units"""
            return model.stored[storage_tecs, h] <= model.energy_capacity[storage_tecs]

        def storage_charging_capacity_constraint_rule(model, h, storage_tecs):
            """Constraint on the capacity with hourly charging relationship of storage. Energy entering the battery
            during one hour cannot exceed the charging capacity."""
            return model.storage[storage_tecs, h] <= model.charging_capacity[storage_tecs]

        def hydrogen_discharge_constraint_rule(model):
            """Constraint on discharge capacity of hydrogen. This is a bit ad hoc, based on discussions with Marie-Alix,
            and some extrapolations for the future capacity of hydrogen.
            We know that for an energy capacity of 3 TWh, we have a maximum injection capacity of 26 GW. So we adapt this value
            to the new energy capacity."""
            return model.capacity["hydrogen"] <= model.energy_capacity['hydrogen'] * 26 / 3000

        def hydrogen_charge_constraint_rule(model):
            """Constraint on charging capacity of hydrogen. This is a bit ad hoc, based on discussions with Marie-Alix,
            and some extrapolations for the future capacity of hydrogen.
            We know that for an energy capacity of 3 TWh, we have a maximum injection capacity of 6.4 GW. So we adapt this value
            to the new energy capacity."""
            return model.charging_capacity["hydrogen"] <= model.energy_capacity['hydrogen'] * 6.4 / 3000

        def phs_charging_constraint_rule(model):
            """We model a constraint on the charging capacity of PHS. Indeed, since we only have a CAPEX associated with the discharging
            capacity, there is no limit to the discharging capacity. The only constraint is that the charging capacity
            should be lower than the discharging capacity. We impose here something slightly more constraining for PHS. The value
            is based on data from annex in RTE (p.898), where we calculate the ratio between the projected charging and
            discharging capacity."""
            return model.charging_capacity['phs'] <= model.capacity['phs'] * 0.91

        def battery_capacity_constraint_rule(model, battery):
            """Constraint on battery's capacity: battery charging capacity equals battery discharging capacity."""
            # TODO: check that the constraint is ok: charging capacity = capacity ?
            return model.charging_capacity[battery] == model.capacity[battery]

        def storage_charging_discharging_constraint_rule(model, storage_tec):
            """Constraint to limit charging capacity to be lower than discharging capacity"""
            return model.charging_capacity[storage_tec] <= model.capacity[storage_tec]

        def methanization_constraint_rule(model, y):
            """Constraint on methanization. The annual power production from methanization is limited to a certain amount."""
            gene_biogas = sum(model.gene['methanization', hour] for hour in range(8760*y,8760*(y+1)-1))
            return gene_biogas <= self.biomass_potential["methanization"] * 1000  # max biogas yearly energy expressed in TWh

        def pyrogazification_constraint_rule(model, y):
            """Constraint on pyrogazification. The annual power production from pyro is limited to a certain amount."""
            gene_pyro = sum(model.gene['pyrogazification', hour] for hour in range(8760*y,8760*(y+1)-1))
            return gene_pyro <= self.biomass_potential["pyrogazification"] * 1000  # max pyro yearly energy expressed in TWh

        def geothermal_constraint_rule(model, y):
            """Constraint on geothermal potential in TWh."""
            gene_geothermal = sum(model.gene['geothermal', hour] for hour in range(8760*y,8760*(y+1)-1))
            return gene_geothermal <= self.district_heating_potential["geothermal"] * 1000

        def central_wood_constraint_rule(model, y):
            """Constraint on central wood potential in TWh."""
            gene_central_wood = sum(model.gene['central_wood_boiler', hour] for hour in range(8760*y,8760*(y+1)-1))
            return gene_central_wood <= self.district_heating_potential["central_wood_boiler"] * 1000

        def uiom_constraint_rule(model, y):
            """Constraint on UIOM potential in TWh."""
            gene_uiom = sum(model.gene['uiom', hour] for hour in range(8760*y,8760*(y+1)-1))
            return gene_uiom <= self.district_heating_potential["uiom"] * 1000

        def reserves_constraint_rule(model, h):
            """Constraint on frr reserves"""
            res_req = sum(self.epsilon[vre] * model.capacity[vre] for vre in model.vre)
            load_req = self.elec_demand[h] * self.miscellaneous['load_uncertainty'] * (1 + self.miscellaneous['delta'])
            return sum(model.reserve[frr, h] for frr in model.frr) == res_req + load_req

        def hydrogen_balance_constraint_rule(model, h):
            """Constraint on hydrogen's balance. Hydrogen production must satisfy CCGT-H2 plants and H2 demand."""
            gene_e_h = model.gene['electrolysis', h] + model.gene['hydrogen', h]
            dem_sto = model.gene['h2_ccgt', h] / self.conversion_efficiency['h2_ccgt'] + self.H2_demand[h] + \
                      model.storage[
                          'hydrogen', h]
            return gene_e_h == dem_sto

        def methane_balance_constraint_rule(model, h):
            """Constraint on methane's balance. Methane production must satisfy CCGT and OCGT plants, CH4 demand and district heating demand for gas."""
            gene_methane = model.gene['methanation', h] + model.gene['methanization', h] + \
                           model.gene['pyrogazification', h] + model.gene['methane', h] + model.gene["natural_gas", h]
            dem_sto = model.gene['ocgt', h] / self.conversion_efficiency['ocgt'] + model.gene['ccgt', h] / \
                      self.conversion_efficiency['ccgt'] + model.gene['central_gas_boiler', h] + self.CH4_demand[h] + model.storage['methane', h]
            return gene_methane == dem_sto

        def district_heating_balance_constraint_rule(model, h):
            """Constraint on district heating balance. District heating demand can be satisfied either by geothermal energy,
            wood biomass, central gas boiler, or storage."""
            gene_DH = sum(model.gene[tec, h] * self.conversion_efficiency[tec] for tec in model.district_heating_balance)
            return gene_DH >= self.district_heating_demand[h]

        def electricity_adequacy_constraint_rule(model, h):
            """Constraint for supply/demand electricity relation'"""
            storage = sum(model.storage[str, h] for str in model.str_elec)  # need in electricity storage
            gene_from_elec = model.gene['electrolysis', h] / self.conversion_efficiency['electrolysis'] + model.gene[
                'methanation', h] / self.conversion_efficiency[
                                 'methanation']  # technologies using electricity for conversion
            prod_elec = sum(model.gene[balance, h] for balance in model.elec_balance)
            return prod_elec >= (
                    self.elec_demand[h] + storage + gene_from_elec)

        def ramping_nuclear_up_constraint_rule(model, h):
            """Constraint setting an upper ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', h] - model.gene['nuclear', previous_h] + model.reserve['nuclear', h] - \
                   model.reserve['nuclear', previous_h] <= \
                   self.hourly_ramping_nuclear * model.capacity['nuclear']

        def ramping_nuclear_down_constraint_rule(model, h):
            """Constraint setting a lower ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', previous_h] - model.gene['nuclear', h] + model.reserve['nuclear', previous_h] - \
                   model.reserve['nuclear', h] <= \
                   self.hourly_ramping_nuclear * model.capacity['nuclear']

        def methanation_CO2_constraint_rule(model, y):
            """Constraint on CO2 balance from methanization, summing over all hours of the year"""
            return sum(model.gene['methanation', h] for h in range(8760*y,8760*(y+1)-1)) / self.conversion_efficiency['methanation'] <= (
                    sum(model.gene['methanization', h] for h in range(8760*y,8760*(y+1)-1)) * self.miscellaneous[
                'percentage_co2_from_methanization']
            )

        def carbon_budget_constraint_rule(model, y):
            """Constraint on carbon budget in MtCO2."""
            # TODO: vérifier la valeur utilisée pour l'intensité carbone du fioul
            return sum(model.gene["natural_gas", h] for h in range(8760*y,8760*(y+1)-1)) * 0.2295 / 1000 + self.oil_consumption * 0.324 / 1000 <= self.carbon_budget


        self.model.generation_vre_constraint = \
            Constraint(self.model.h, self.model.vre, rule=generation_vre_constraint_rule)

        self.model.generation_nuclear_constraint = Constraint(self.model.years, rule=generation_nuclear_constraint_rule)

        self.model.generation_nuclear_hourly_constraint = Constraint(self.model.h, rule=generation_nuclear_constraint_hourly_rule)

        self.model.generation_capacity_constraint = \
            Constraint(self.model.h, self.model.tec, rule=generation_capacity_constraint_rule)

        self.model.battery_1_capacity_constraint = Constraint(rule=battery1_capacity_constraint_rule)

        self.model.battery_4_capacity_constraint = Constraint(rule=battery4_capacity_constraint_rule)

        self.model.frr_capacity_constraint = Constraint(self.model.h, self.model.frr, rule=frr_capacity_constraint_rule)

        self.model.storing_constraint = Constraint(self.model.h, self.model.str, rule=storing_constraint_rule)

        self.model.storage_constraint = Constraint(self.model.str, rule=storage_first_last_constraint_rule)

        self.model.lake_reserve_constraint = Constraint(self.model.months, rule=lake_reserve_constraint_rule)

        self.model.stored_capacity_constraint = Constraint(self.model.h, self.model.str,
                                                           rule=stored_capacity_constraint)

        self.model.storage_capacity_1_constraint = \
            Constraint(self.model.h, self.model.str, rule=storage_charging_capacity_constraint_rule)

        # TODO: new rule
        self.model.hydrogen_discharge_constraint = Constraint(rule=hydrogen_discharge_constraint_rule)

        self.model.hydrogen_charge_constraint = Constraint(rule=hydrogen_charge_constraint_rule)

        self.model.phs_charging_constraint = Constraint(rule=phs_charging_constraint_rule)

        self.model.battery_capacity_constraint = Constraint(self.model.battery, rule=battery_capacity_constraint_rule)

        self.model.storage_charging_discharging_constraint = \
            Constraint(self.model.str, rule=storage_charging_discharging_constraint_rule)

        self.model.biogas_constraint = Constraint(self.model.years, rule=methanization_constraint_rule)

        self.model.pyrogazification_constraint = Constraint(self.model.years, rule=pyrogazification_constraint_rule)

        self.model.geothermal_constraint = Constraint(self.model.years, rule=geothermal_constraint_rule)

        self.model.central_wood_constraint = Constraint(self.model.years, rule=central_wood_constraint_rule)

        self.model.uiom_constraint_rule = Constraint(self.model.years, rule=uiom_constraint_rule)

        self.model.ramping_nuclear_up_constraint = Constraint(self.model.h, rule=ramping_nuclear_up_constraint_rule)

        self.model.ramping_nuclear_down_constraint = Constraint(self.model.h, rule=ramping_nuclear_down_constraint_rule)

        self.model.methanation_constraint = Constraint(self.model.years, rule=methanation_CO2_constraint_rule)

        self.model.reserves_constraint = Constraint(self.model.h, rule=reserves_constraint_rule)

        self.model.hydrogen_balance_constraint = Constraint(self.model.h, rule=hydrogen_balance_constraint_rule)

        self.model.methane_balance_constraint = Constraint(self.model.h, rule=methane_balance_constraint_rule)

        self.model.district_heating_balance_constraint = Constraint(self.model.h, rule=district_heating_balance_constraint_rule)

        self.model.electricity_adequacy_constraint = Constraint(self.model.h, rule=electricity_adequacy_constraint_rule)

        if self.carbon_constraint:  # on ajoute la contrainte carbone
            self.model.carbon_budget_constraint = Constraint(self.model.years, rule=carbon_budget_constraint_rule)

    def define_objective(self):
        def objective_rule(model):
            """Objective value in 10**3 M€, or 1e9€"""
            return (sum(
                (model.capacity[tec] - self.existing_capacity[tec]) * self.annuities[tec] for tec in
                model.tec)
                    + sum(
                        (model.energy_capacity[storage_tecs] - self.existing_energy_capacity[storage_tecs]) *
                        self.storage_annuities[
                            storage_tecs] for storage_tecs in model.str)
                    # + sum(
                    #     (model.charging_capacity[storage_tecs] - self.existing_charging_capacity[storage_tecs]) *
                    #     self.charging_capex[
                    #         storage_tecs] * self.nb_years for storage_tecs in model.str)
                    + sum(model.capacity[tec] * self.fOM[tec] for tec in model.tec)
                    # + sum(
                    #     model.charging_capacity[storage_tecs] * self.charging_opex[storage_tecs] * self.nb_years
                    #     for storage_tecs in model.str)
                    + sum(sum(model.gene[tec, h] * self.vOM[tec] for h in model.h) for tec in model.tec) / self.nb_years
                    + self.oil_consumption * self.vOM["oil"] / self.nb_years
                    + (self.wood_consumption + sum(model.gene["central_wood_boiler", h] for h in model.h)) * self.vOM["wood"] / self.nb_years  # we add variable costs from wood and fuel
                    ) / 1000

        # Creation of the objective -> Cost
        self.model.objective = Objective(rule=objective_rule)

    def build_model(self):
        self.define_sets()
        self.define_other_demand()
        self.define_variables()
        self.fix_values()
        self.define_constraints()
        self.define_objective()

    def solve(self, solver_name, infeasible_value=1000):
        """Attention au choix de la infeasible_value: c'est la valeur que l'on donne lorsque le problème n'est pas solvable."""
        self.opt = SolverFactory(solver_name)
        self.logger.info("Solving EOLES model using %s", self.opt.name)
        # self.solver_results = self.opt.solve(self.model,
        #                                      options={'Presolve': 2, 'LogFile': self.path + "/logfile_" + self.name})

        self.solver_results = self.opt.solve(self.model,
                                             options={'threads': 4,
                                                      'method': 2, # barrier
                                                      'crossover': 0,
                                                      'BarConvTol': 1.e-6,
                                                       'Seed': 123,
                                                       'AggFill': 0,
                                                       'PreDual': 0,
                                                       'GURO_PAR_BARDENSETHRESH': 200,
                                                      'LogFile': self.path + "/logfile_" + self.name})

        status = self.solver_results["Solver"][0]["Status"]
        termination_condition = self.solver_results["Solver"][0]["Termination condition"]

        if status == "ok" and termination_condition == "optimal":
            self.logger.info("Optimization successful")
            self.extract_optimisation_results()
        elif status == "warning" and termination_condition == "other":
            self.logger.warning(
                "WARNING! Optimization might be sub-optimal. Writing output anyway"
            )
            self.extract_optimisation_results()
        else:
            self.logger.error(
                "Optimisation failed with status %s and terminal condition %s"
                % (status, termination_condition)
            )
            # self.objective = np.nan
            self.objective = infeasible_value
        return self.solver_results, status, termination_condition

    def extract_optimisation_results(self):
        """

        :param m: ModelEOLES
        :return:
        """
        # get value of objective function
        self.objective = self.solver_results["Problem"][0]["Upper bound"]
        self.technical_cost, self.emissions = get_technical_cost(self.model, self.objective, self.anticipated_scc, self.oil_consumption, self.nb_years)
        self.hourly_generation = extract_hourly_generation(self.model, elec_demand=self.elec_demand,  CH4_demand=list(self.CH4_demand.values()),
                                                           H2_demand=list(self.H2_demand.values()), conversion_efficiency=self.conversion_efficiency,
                                                           hourly_heat_elec=self.hourly_heat_elec, hourly_heat_gas=self.hourly_heat_gas)
        self.gas_carbon_content, self.dh_carbon_content, self.heat_elec_carbon_content, self.heat_elec_carbon_content_day = \
            get_carbon_content(self.hourly_generation, self.conversion_efficiency)
        self.peak_electricity_load_info = extract_peak_load(self.hourly_generation, self.conversion_efficiency, self.input_years)
        self.peak_heat_load_info = extract_peak_heat_load(self.hourly_generation, self.input_years)
        self.spot_price = extract_spot_price(self.model, self.last_hour)
        self.carbon_value = extract_carbon_value(self.model, self.carbon_constraint, self.anticipated_scc)
        self.capacities = extract_capacities(self.model)
        self.energy_capacity = extract_energy_capacity(self.model)
        self.charging_capacity = extract_charging_capacity(self.model)
        self.electricity_generation = extract_supply_elec(self.model, self.nb_years)
        self.primary_generation = extract_primary_gene(self.model, self.nb_years)
        self.CH4_to_power_generation = extract_CH4_to_power(self.model, self.conversion_efficiency, self.nb_years)
        self.H2_to_power_generation = extract_H2_to_power(self.model, self.conversion_efficiency, self.nb_years)
        self.power_to_CH4_generation = extract_power_to_CH4(self.model, self.conversion_efficiency, self.nb_years)
        self.power_to_H2_generation = extract_power_to_H2(self.model, self.conversion_efficiency, self.nb_years)

        self.new_capacity_annualized_costs, self.new_energy_capacity_annualized_costs = \
            extract_annualized_costs_investment_new_capa(self.capacities, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                                         self.storage_annuities, self.fOM)

        # self.use_elec = extract_use_elec(self.model, self.nb_years, self.miscellaneous)
        self.transport_distribution_cost = transportation_distribution_cost(self.model, self.prediction_transport_and_distrib_annuity)
        self.summary, self.generation_per_technology, \
        self.lcoe_per_tec = extract_summary(self.model, self.elec_demand, self.H2_demand, self.CH4_demand,
                                            self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                            self.storage_annuities, self.fOM, self.vOM, self.conversion_efficiency,
                                            self.existing_annualized_costs_elec, self.existing_annualized_costs_CH4,
                                            self.existing_annualized_costs_CH4_naturalgas, self.existing_annualized_costs_CH4_biogas,
                                            self.existing_annualized_costs_H2, self.transport_distribution_cost,
                                            self.anticipated_scc, self.nb_years, self.carbon_constraint)
        self.new_capacity_annualized_costs_nofOM, self.new_energy_capacity_annualized_costs_nofOM = \
            extract_annualized_costs_investment_new_capa_nofOM(self.capacities, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                                         self.storage_annuities)  # pd.Series
        self.functionment_cost = extract_functionment_cost(self.model, self.capacities, self.fOM, self.vOM,
                                                           pd.Series(self.generation_per_technology) * 1000, self.oil_consumption, self.wood_consumption,
                                                           self.anticipated_scc, self.actual_scc, carbon_constraint=self.carbon_constraint,
                                                           nb_years=self.nb_years)  # pd.Series
        self.results = {'objective': self.objective, 'summary': self.summary,
                        'hourly_generation': self.hourly_generation,
                        'capacities': self.capacities, 'energy_capacity': self.energy_capacity,
                        'supply_elec': self.electricity_generation, 'primary_generation': self.primary_generation}



def read_hourly_data(config, year, scenario='Reference', method="valentin", calibration=False, hourly_heat_elec=None):
    """Reads data defined at the hourly scale"""
    load_factors = get_pandas(config["load_factors"],
                              lambda x: pd.read_csv(x, index_col=[0, 1], header=None).squeeze("columns"))
    demand = get_pandas(config["demand"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW

    demand_no_residential = process_RTE_demand(config, year, demand, scenario, method=method, calibration=calibration, hourly_residential_heating_RTE=hourly_heat_elec)

    lake_inflows = get_pandas(config["lake_inflows"],
                              lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh

    o = dict()
    o["load_factors"] = load_factors
    o["demand"] = demand_no_residential
    o["lake_inflows"] = lake_inflows
    return o


def read_technology_data(config, year):
    """Read technology data (capex, opex, capacity potential, etc...)
        config: json file
            Includes paths to files
        year: int
            Year to get capex."""
    epsilon = get_pandas(config["epsilon"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    existing_capacity = get_pandas(config["existing_capacity"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_charging_capacity = get_pandas(config["existing_charging_capacity"],
                                            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_energy_capacity = get_pandas(config["existing_energy_capacity"],
                                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_capacity = get_pandas(config["maximum_capacity"],
                                  lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_charging_capacity = get_pandas(config["maximum_charging_capacity"],
                                           lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_energy_capacity = get_pandas(config["maximum_energy_capacity"],
                                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_capacities = get_pandas(config["fix_capacities"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_charging_capacities = get_pandas(config["fix_charging_capacities"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_energy_capacities = get_pandas(config["fix_energy_capacities"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh
    lifetime = get_pandas(config["lifetime"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    construction_time = get_pandas(config["construction_time"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    capex = get_pandas(config["capex"],
                       lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    capex = capex[[str(year)]].squeeze()  # get capex for year of interest
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    storage_capex = storage_capex[[str(year)]].squeeze()  # get storage capex for year of interest
    fOM = get_pandas(config["fOM"], lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW/year
    fOM = fOM[[str(year)]].squeeze()  # get fOM for year of interest
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GWh
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    conversion_efficiency = get_pandas(config["conversion_efficiency"],
                                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    prediction_transport_and_distrib_annuity = get_pandas(config["prediction_transport_distribution"],
                               lambda x: pd.read_csv(x, index_col=0).squeeze("columns"))
    prediction_transport_offshore_annuity = get_pandas(config["prediction_transport_offshore"],
                               lambda x: pd.read_csv(x, index_col=0).squeeze("columns"))
    # biomass_potential = get_pandas(config["biomass_potential"],
    #                                lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    # biomass_potential = biomass_potential[[str(year)]].squeeze()  # get storage capex for year of interest

    o = dict()
    o["epsilon"] = epsilon
    o["existing_capacity"] = existing_capacity
    o["existing_charging_capacity"] = existing_charging_capacity
    o["existing_energy_capacity"] = existing_energy_capacity
    o["maximum_capacity"] = maximum_capacity
    o["maximum_charging_capacity"] = maximum_charging_capacity
    o["maximum_energy_capacity"] = maximum_energy_capacity
    o["fix_capacities"] = fix_capacities
    o["fix_charging_capacities"] = fix_charging_capacities
    o["fix_energy_capacities"] = fix_energy_capacities
    o["lifetime"] = lifetime
    o["construction_time"] = construction_time
    o["capex"] = capex
    o["storage_capex"] = storage_capex
    o["fOM"] = fOM
    o["vOM"] = vOM
    o["eta_in"] = eta_in
    o["eta_out"] = eta_out
    o["conversion_efficiency"] = conversion_efficiency
    o["miscellaneous"] = miscellaneous
    o["prediction_transport_and_distrib_annuity"] = prediction_transport_and_distrib_annuity
    o["prediction_transport_offshore_annuity"] = prediction_transport_offshore_annuity
    # o["biomass_potential"] = biomass_potential
    return o


def read_annual_data(config, year):
    """Read annual demand data (H2, energy prices)
        config: json file
            Includes paths to files
        year: int
            Year to get capex."""
    demand_H2_timesteps = get_pandas(config["demand_H2_timesteps"],
                                     lambda x: pd.read_csv(x, index_col=0).squeeze())
    demand_H2_RTE = demand_H2_timesteps[year]  # TWh

    biomass_potential = get_pandas(config["biomass_potential"],
                                   lambda x: pd.read_csv(x, index_col=[0, 1]))  # 1e6€/GW
    biomass_potential = biomass_potential.loc[config["biomass_potential_scenario"]]  # we select the desired ADEME scenario
    biomass_potential = biomass_potential[[str(year)]].squeeze()  # get specific potential for year of interest

    district_heating_potential = get_pandas(config["district_heating_potential"],
                                   lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    district_heating_potential = district_heating_potential[[str(year)]].squeeze()  # get storage capex for year of interest

    energy_prices = get_pandas(config["energy_prices"],
                               lambda x: pd.read_csv(x, index_col=0))  # €/MWh
    energy_prices = energy_prices[[str(year)]].squeeze()  # get storage capex for year of interest
    carbon_budget_timesteps = get_pandas(config["carbon_budget"], lambda x: pd.read_csv(x, index_col=0).squeeze())
    carbon_budget = carbon_budget_timesteps[year]

    o = dict()
    o["demand_H2_RTE"] = demand_H2_RTE * 1e3  # GWh
    o["energy_prices"] = energy_prices  # € / MWh
    o["carbon_budget"] = carbon_budget  # MtCO2eq
    o["biomass_potential"] = biomass_potential
    o["district_heating_potential"] = district_heating_potential
    return o


def read_input_static(config, year):
    """Read static data"""
    epsilon = get_pandas(config["epsilon"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    existing_capacity = get_pandas(config["existing_capacity"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_charging_capacity = get_pandas(config["existing_charging_capacity"],
                                            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_energy_capacity = get_pandas(config["existing_energy_capacity"],
                                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_capacity = get_pandas(config["maximum_capacity"],
                                  lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_charging_capacity = get_pandas(config["maximum_charging_capacity"],
                                           lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_energy_capacity = get_pandas(config["maximum_energy_capacity"],
                                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_capacities = get_pandas(config["fix_capacities"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    lifetime = get_pandas(config["lifetime"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    construction_time = get_pandas(config["construction_time"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    capex = get_pandas(config["capex"],
                       lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    capex = capex[[str(year)]].squeeze()  # get capex for year of interest
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    storage_capex = storage_capex[[str(year)]].squeeze()  # get storage capex for year of interest
    fOM = get_pandas(config["fOM"], lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW/year
    fOM = fOM[[str(year)]].squeeze()  # get fOM for year of interest
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GWh
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    conversion_efficiency = get_pandas(config["conversion_efficiency"],
                                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    biomass_potential = get_pandas(config["biomass_potential"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    biomass_potential = biomass_potential[[str(year)]].squeeze()  # get storage capex for year of interest
    demand_H2_timesteps = get_pandas(config["demand_H2_timesteps"],
                                     lambda x: pd.read_csv(x, index_col=0).squeeze())
    demand_H2_RTE = demand_H2_timesteps[year]  # TWh

    o = dict()
    o["epsilon"] = epsilon
    o["existing_capacity"] = existing_capacity
    o["existing_charging_capacity"] = existing_charging_capacity
    o["existing_energy_capacity"] = existing_energy_capacity
    o["maximum_capacity"] = maximum_capacity
    o["maximum_charging_capacity"] = maximum_charging_capacity
    o["maximum_energy_capacity"] = maximum_energy_capacity
    o["fix_capacities"] = fix_capacities
    o["lifetime"] = lifetime
    o["construction_time"] = construction_time
    o["capex"] = capex
    o["storage_capex"] = storage_capex
    o["fOM"] = fOM
    o["vOM"] = vOM
    o["eta_in"] = eta_in
    o["eta_out"] = eta_out
    o["conversion_efficiency"] = conversion_efficiency
    o["miscellaneous"] = miscellaneous
    o["biomass_potential"] = biomass_potential
    o["demand_H2_RTE"] = demand_H2_RTE * 1e3  # GWh
    return o


def extract_summary(model, elec_demand, H2_demand, CH4_demand, existing_capacity, existing_energy_capacity, annuities,
                    storage_annuities, fOM, vOM, conversion_efficiency, existing_annualized_costs_elec,
                    existing_annualized_costs_CH4, existing_annualized_costs_CH4_naturalgas, existing_annualized_costs_CH4_biogas,
                    existing_annualized_costs_H2, transportation_distribution_cost,
                    anticipated_scc, nb_years, carbon_constraint):
    """This function compiles different general statistics of the electricity mix, including in particular LCOE."""
    summary = {}  # final dictionary for output
    elec_demand_tot = sum(elec_demand[hour] for hour in model.h) / 1000  # electricity demand in TWh
    H2_demand_tot = sum(H2_demand[hour] for hour in model.h) / 1000  # H2 demand in TWh
    CH4_demand_tot = sum(CH4_demand[hour] for hour in model.h) / 1000  # CH4 demand in TWh

    elec_spot_price = [-1e6 * model.dual[model.electricity_adequacy_constraint[h]] for h in
                       model.h]  # 1e3€/GWh = €/MWh
    CH4_spot_price = [1e6 * model.dual[model.methane_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh = €/MWh
    H2_spot_price = [1e6 * model.dual[model.hydrogen_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh = €/MWh
    gene_elec = [sum(value(model.gene[tec, hour]) for tec in model.elec_gene) for hour in model.h]
    storage_elec = [sum(value(model.gene[tec, hour]) for tec in model.str_elec) for hour in model.h]

    weighted_elec_price_demand = sum([elec_spot_price[h] * elec_demand[h] for h in model.h]) / (
            elec_demand_tot * 1e3)  # €/MWh
    summary["weighted_elec_price_demand"] = weighted_elec_price_demand
    weighted_CH4_price_demand = sum([CH4_spot_price[h] * CH4_demand[h] for h in model.h]) / (
            CH4_demand_tot * 1e3)  # €/MWh
    summary["weighted_CH4_price_demand"] = weighted_CH4_price_demand
    weighted_H2_price_demand = sum([H2_spot_price[h] * H2_demand[h] for h in model.h]) / (
            H2_demand_tot * 1e3)  # €/MWh
    summary["weighted_H2_price_demand"] = weighted_H2_price_demand

    weighted_price_generation = sum([elec_spot_price[h] * gene_elec[h] for h in model.h]) / sum(gene_elec)  # €/MWh
    summary["weighted_price_generation"] = weighted_price_generation

    summary["elec_demand_tot"] = elec_demand_tot
    summary["hydrogen_demand_tot"] = H2_demand_tot
    summary["methane_demand_tot"] = CH4_demand_tot

    # Overall energy generated by the technology in TWh over total considered years
    gene_per_tec = {}
    for tec in model.tec:
        gene_per_tec[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000  # TWh

    # summary.update(gene_per_tec)

    sumgene_elec = sum(gene_per_tec[tec] for tec in model.elec_gene) + gene_per_tec["ocgt"] + gene_per_tec["ccgt"] + \
                   gene_per_tec["h2_ccgt"]  # production in TWh
    summary["sumgene_elec"] = sumgene_elec
    sumgene_CH4 = sum(gene_per_tec[tec] for tec in model.CH4_gene) # production in TWh
    summary["sumgene_CH4"] = sumgene_CH4
    sumgene_H2 = sum(gene_per_tec[tec] for tec in model.H2_gene) # production in TWh
    summary["sumgene_H2"] = sumgene_H2

    # LCOE per technology
    lcoe_per_tec = {}
    lcoe_elec_gene = calculate_LCOE_gene_tec(model.elec_gene, model, annuities, fOM, vOM, nb_years,
                                             gene_per_tec)  # € / MWh-e
    lcoe_elec_conv_CH4 = calculate_LCOE_conv_tec(["ocgt", "ccgt"], model, annuities, fOM, conversion_efficiency,
                                                 CH4_spot_price, nb_years, gene_per_tec)  # € / MWh-e
    lcoe_elec_conv_H2 = calculate_LCOE_conv_tec(["h2_ccgt"], model, annuities, fOM, conversion_efficiency,
                                                H2_spot_price, nb_years, gene_per_tec)  # € / MWh-e
    lcoe_gas_gene = calculate_LCOE_gene_tec(model.gas_gene, model, annuities, fOM, vOM, nb_years,
                                            gene_per_tec)  # € / MWh-th
    lcoe_per_tec.update(lcoe_elec_gene)
    lcoe_per_tec.update(lcoe_elec_conv_CH4)
    lcoe_per_tec.update(lcoe_elec_conv_H2)
    lcoe_per_tec.update(lcoe_gas_gene)

    G2P_bought = sum(CH4_spot_price[hour] * (
            value(model.gene["ocgt", hour]) / conversion_efficiency['ocgt'] + value(model.gene["ccgt", hour]) /
            conversion_efficiency['ccgt'])
                     for hour in model.h) / 1e3 + sum(H2_spot_price[hour] * (
            value(model.gene["h2_ccgt", hour]) / conversion_efficiency['h2_ccgt']) for hour in
                                                      model.h) / 1e3  # 1e6€ car l'objectif du modèle est en 1e9 €

    P2G_CH4_bought = sum(elec_spot_price[hour] * sum(
            value(model.gene[tec, hour])/conversion_efficiency[tec] for tec in model.from_elec_to_CH4) for hour in model.h) / 1e3
    P2G_H2_bought = sum(elec_spot_price[hour] * sum(
            value(model.gene[tec, hour])/conversion_efficiency[tec] for tec in model.from_elec_to_H2) for hour in model.h) / 1e3

    # We calculate the costs associated to functioning of each system (elec, CH4, gas)
    costs_elec, costs_CH4, cost_CH4_naturalgas, cost_CH4_biogas, costs_H2 = compute_costs(model, annuities, fOM, vOM, storage_annuities, gene_per_tec,
                                                                            existing_capacity, existing_energy_capacity,
                  existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_CH4_naturalgas,
                                      existing_annualized_costs_CH4_biogas, existing_annualized_costs_H2, nb_years)  # 1e6 €

    # print(costs_elec, costs_CH4, costs_H2)
    # We first calculate LCOE by using total costs.
    lcoe_elec, lcoe_CH4, lcoe_CH4_naturalgas, lcoe_CH4_biogas, lcoe_H2 = \
        compute_lcoe(costs_elec, costs_CH4, cost_CH4_naturalgas, cost_CH4_biogas, costs_H2, G2P_bought, P2G_CH4_bought,
                     P2G_H2_bought, sumgene_elec, sumgene_CH4, sumgene_H2)
    summary["lcoe_elec"] = lcoe_elec
    summary["lcoe_CH4"] = lcoe_CH4
    summary["lcoe_CH4_naturalgas"] = lcoe_CH4_naturalgas
    summary["lcoe_CH4_biogas"] = lcoe_CH4_biogas
    assert math.isclose(lcoe_CH4, lcoe_CH4_naturalgas + lcoe_CH4_biogas), "Problem when estimating CH4 LCOE."
    summary["lcoe_H2"] = lcoe_H2

    # We now calculate ratios to assign the costs depending on the part of those costs used to meet final demand,
    # or to meet other vectors demand.
    # Option 1: based on volumetric assumptions
    lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume = \
        compute_lcoe_volumetric(model, gene_per_tec, conversion_efficiency, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot)
    summary["lcoe_elec_volume"], summary["lcoe_CH4_volume"], summary["lcoe_H2_volume"] = \
        lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume

    # Option 2: based on value assumptions (we weight each volume by the hourly price
    lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value = \
        compute_lcoe_value(model, conversion_efficiency, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot,
                       elec_demand, CH4_demand, H2_demand, elec_spot_price, CH4_spot_price, H2_spot_price) # €/MWh
    summary["lcoe_elec_value"], summary["lcoe_CH4_value"], summary["lcoe_H2_value"] = \
        lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value

    if not carbon_constraint:
        # We compile CH4 LCOE without SCC. This is needed for the calibration and estimation of gas prices.
        costs_elec_noSCC, costs_CH4_noSCC, cost_CH4_naturalgas_noSCC, cost_CH4_biogas_noSCC,  costs_H2_noSCC = \
            compute_costs_noSCC(model, annuities, fOM, vOM, storage_annuities, anticipated_scc, gene_per_tec,
                                existing_capacity, existing_energy_capacity, existing_annualized_costs_elec,
                                existing_annualized_costs_CH4, existing_annualized_costs_CH4_naturalgas, existing_annualized_costs_CH4_biogas,
                                existing_annualized_costs_H2, nb_years)  # 1e6 €

        lcoe_elec_noSCC, lcoe_CH4_noSCC, lcoe_CH4_naturalgas_noSCC, lcoe_CH4_biogas_noSCC, lcoe_H2_noSCC = \
            compute_lcoe(costs_elec_noSCC, costs_CH4_noSCC, cost_CH4_naturalgas_noSCC, cost_CH4_biogas_noSCC, costs_H2_noSCC,
                         G2P_bought, P2G_CH4_bought, P2G_H2_bought, sumgene_elec, sumgene_CH4, sumgene_H2)

        lcoe_elec_volume_noSCC, lcoe_CH4_volume_noSCC, lcoe_H2_volume_noSCC = \
            compute_lcoe_volumetric(model, gene_per_tec, conversion_efficiency, costs_elec_noSCC, costs_CH4_noSCC, costs_H2_noSCC, elec_demand_tot, CH4_demand_tot, H2_demand_tot)

    else:  # no difference because SCC = 0 in that case
        lcoe_elec_noSCC, lcoe_CH4_noSCC, lcoe_CH4_naturalgas_noSCC, lcoe_CH4_biogas_noSCC, lcoe_H2_noSCC = lcoe_elec, lcoe_CH4, lcoe_CH4_naturalgas, lcoe_CH4_biogas, lcoe_H2
        lcoe_elec_volume_noSCC, lcoe_CH4_volume_noSCC, lcoe_H2_volume_noSCC = lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume

    summary["lcoe_CH4_noSCC"], summary["lcoe_CH4_volume_noSCC"] = lcoe_CH4_noSCC, lcoe_CH4_volume_noSCC
    summary["lcoe_CH4_naturalgas_noSCC"], summary["lcoe_CH4_biogas_noSCC"] = lcoe_CH4_naturalgas_noSCC, lcoe_CH4_biogas_noSCC
    assert math.isclose(lcoe_CH4_noSCC, lcoe_CH4_naturalgas_noSCC + lcoe_CH4_biogas_noSCC), "Problem when estimating CH4 noSCC LCOE."

    # Estimation of transportation and distribution costs
    transport_and_distrib_lcoe = (transportation_distribution_cost * 1000 * nb_years) / elec_demand_tot  # € / yr / MWh

    summary["transport_and_distrib_lcoe"] = transport_and_distrib_lcoe

    summary_df = pd.Series(summary)
    return summary_df, gene_per_tec, lcoe_per_tec


def compute_costs(model, annuities, fOM, vOM, storage_annuities, gene_per_tec, existing_capacity, existing_energy_capacity,
                  existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_CH4_naturalgas,
                  existing_annualized_costs_CH4_biogas, existing_annualized_costs_H2, nb_years):
    costs_elec = existing_annualized_costs_elec + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.elec_balance) + \
                 sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                         str] * nb_years for str in model.str_elec) # 1e6 €

    costs_CH4 = existing_annualized_costs_CH4 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.CH4_balance) + \
                sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                        str] * nb_years for str in model.str_CH4) # 1e6 €

    cost_CH4_naturalgas = existing_annualized_costs_CH4_naturalgas + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.CH4_balance_historic) + \
                sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                        str] * nb_years for str in model.str_CH4) # 1e6 €

    cost_CH4_biogas = existing_annualized_costs_CH4_biogas + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.CH4_balance_biogas)  # 1e6 €

    costs_H2 = existing_annualized_costs_H2 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.H2_balance) + \
               sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                       str] * nb_years for str in model.str_H2) # 1e6 €

    return costs_elec, costs_CH4, cost_CH4_naturalgas, cost_CH4_biogas, costs_H2


def compute_costs_noSCC(model, annuities, fOM, vOM, storage_annuities, anticipated_scc, gene_per_tec, existing_capacity,
                        existing_energy_capacity, existing_annualized_costs_elec, existing_annualized_costs_CH4,
                        existing_annualized_costs_CH4_naturalgas, existing_annualized_costs_CH4_biogas,
                            existing_annualized_costs_H2, nb_years):
    """Same as compute_costs, but only includes technical costs, and no SCC."""
    new_vOM = vOM.copy()
    new_vOM.loc["natural_gas"] = update_ngas_cost(new_vOM.loc["natural_gas"], scc=(0 - anticipated_scc),
                                                  emission_rate=0.2295)  # we go back to initial cost without the SCC
    costs_elec = existing_annualized_costs_elec + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[
            tec] * new_vOM[tec] * 1000 for tec in model.elec_balance) + \
                 sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                         str] * nb_years for str in model.str_elec)  # 1e6 €

    costs_CH4 = existing_annualized_costs_CH4 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[
            tec] * new_vOM[tec] * 1000 for tec in model.CH4_balance) + \
                sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                        str] * nb_years for str in model.str_CH4)  # 1e6 €

    cost_CH4_naturalgas = existing_annualized_costs_CH4_naturalgas + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * new_vOM[tec] * 1000
        for tec in model.CH4_balance_historic) + \
                sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                        str] * nb_years for str in model.str_CH4) # 1e6 €

    cost_CH4_biogas = existing_annualized_costs_CH4_biogas + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * new_vOM[tec] * 1000
        for tec in model.CH4_balance_biogas)  # 1e6 €

    costs_H2 = existing_annualized_costs_H2 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[
            tec] * new_vOM[tec] * 1000 for tec in model.H2_balance) + \
               sum((value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                       str] * nb_years for str in model.str_H2)  # 1e6 €

    return costs_elec, costs_CH4, cost_CH4_naturalgas, cost_CH4_biogas, costs_H2


def compute_lcoe(costs_elec, costs_CH4, cost_CH4_naturalgas, cost_CH4_biogas, costs_H2, G2P_bought, P2G_CH4_bought, P2G_H2_bought, sumgene_elec, sumgene_CH4, sumgene_H2):
    """Compute LCOE by using the costs of buying electricity / CH4 / H2 to work. Parameters sumgene_elec, sumgene_CH4 and
    sumgene_H2 refer to the total production from each system (which can be used either to satisfy final demand, or for
     vector coupling."""
    lcoe_elec = (costs_elec + G2P_bought) / sumgene_elec  # €/MWh
    lcoe_CH4 = (costs_CH4 + P2G_CH4_bought) / sumgene_CH4  # €/MWh
    lcoe_CH4_naturalgas = cost_CH4_naturalgas / sumgene_CH4  # €/MWh
    lcoe_CH4_biogas = (cost_CH4_biogas + P2G_CH4_bought) / sumgene_CH4  # €/MWh
    lcoe_H2 = (costs_H2 + P2G_H2_bought) / sumgene_H2  # €/MWh
    return lcoe_elec, lcoe_CH4, lcoe_CH4_naturalgas, lcoe_CH4_biogas, lcoe_H2


def compute_lcoe_volumetric(model, gene_per_tec, conversion_efficiency, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot):
    """Computes a volumetric LCOE, where costs of each system (respectively, electricity, methane and hydrogen) are distributed across the different
    systems based on volumes (eg, volume of demand versus volume of gas used for the electricity system)."""
    gene_from_CH4_to_elec = sum(gene_per_tec[tec]/conversion_efficiency[tec] for tec in model.from_CH4_to_elec)  # TWh
    gene_from_H2_to_elec = sum(gene_per_tec[tec]/conversion_efficiency[tec] for tec in model.from_H2_to_elec)  # TWh
    gene_from_elec_to_CH4 = sum(gene_per_tec[tec]/conversion_efficiency[tec] for tec in model.from_elec_to_CH4)  # TWh
    gene_from_elec_to_H2 = sum(gene_per_tec[tec]/conversion_efficiency[tec] for tec in model.from_elec_to_H2)  # TWh

    costs_CH4_to_demand = costs_CH4 * CH4_demand_tot / (CH4_demand_tot + gene_from_CH4_to_elec)  # 1e6 €
    costs_CH4_to_elec = costs_CH4 * gene_from_CH4_to_elec / (CH4_demand_tot + gene_from_CH4_to_elec)
    costs_H2_to_demand = costs_H2 * H2_demand_tot / (H2_demand_tot + gene_from_H2_to_elec)
    costs_H2_to_elec = costs_H2 * gene_from_H2_to_elec / (H2_demand_tot + gene_from_H2_to_elec)
    costs_elec_to_demand = costs_elec * elec_demand_tot / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_CH4 = costs_elec * gene_from_elec_to_CH4 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_H2 = costs_elec * gene_from_elec_to_H2 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)

    lcoe_elec_volume = (costs_CH4_to_elec + costs_H2_to_elec + costs_elec_to_demand) / elec_demand_tot  # € / MWh
    lcoe_CH4_volume = (costs_elec_to_CH4 + costs_CH4_to_demand) / CH4_demand_tot  # € / MWh
    lcoe_H2_volume = (costs_elec_to_H2 + costs_H2_to_demand) / H2_demand_tot  # € / MWh
    return lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume


def compute_lcoe_value(model, conversion_efficiency, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot,
                       elec_demand, CH4_demand, H2_demand, elec_spot_price, CH4_spot_price, H2_spot_price):
    total_elec_spot_price = sum(elec_spot_price)
    total_CH4_spot_price = sum(CH4_spot_price)
    total_H2_spot_price = sum(H2_spot_price)
    gene_from_CH4_to_elec_value = sum(
        sum(value(model.gene[tec, hour])/conversion_efficiency[tec] * CH4_spot_price[hour] for hour in model.h) / (1000 * total_CH4_spot_price)
        for tec in model.from_CH4_to_elec)  # TWh
    gene_from_H2_to_elec_value = sum(
        sum(value(model.gene[tec, hour])/conversion_efficiency[tec] * H2_spot_price[hour] for hour in model.h) / (1000 * total_H2_spot_price)
        for tec in model.from_H2_to_elec)  # TWh
    gene_from_elec_to_CH4_value = sum(
        sum(value(model.gene[tec, hour])/conversion_efficiency[tec] * elec_spot_price[hour] for hour in model.h) / (
                1000 * total_elec_spot_price) for tec in model.from_elec_to_CH4)  # TWh
    gene_from_elec_to_H2_value = sum(
        sum(value(model.gene[tec, hour])/conversion_efficiency[tec] * elec_spot_price[hour] for hour in model.h) / (
                1000 * total_elec_spot_price) for tec in model.from_elec_to_H2)  # TWh
    elec_demand_tot_value = sum(elec_demand[hour] * elec_spot_price[hour] for hour in model.h) / (
            1000 * total_elec_spot_price)
    CH4_demand_tot_value = sum(CH4_demand[hour] * CH4_spot_price[hour] for hour in model.h) / (
            1000 * total_CH4_spot_price)
    H2_demand_tot_value = sum(H2_demand[hour] * H2_spot_price[hour] for hour in model.h) / (
            1000 * total_H2_spot_price)

    costs_CH4_to_demand_value = costs_CH4 * CH4_demand_tot_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)  # 1e6 €
    costs_CH4_to_elec_value = costs_CH4 * gene_from_CH4_to_elec_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    costs_H2_to_demand_value = costs_H2 * H2_demand_tot_value / (H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_H2_to_elec_value = costs_H2 * gene_from_H2_to_elec_value / (
            H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_elec_to_demand_value = costs_elec * elec_demand_tot_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_CH4_value = costs_elec * gene_from_elec_to_CH4_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_H2_value = costs_elec * gene_from_elec_to_H2_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)

    lcoe_elec_value = (costs_CH4_to_elec_value + costs_H2_to_elec_value + costs_elec_to_demand_value) / elec_demand_tot  # € / MWh
    lcoe_CH4_value = (costs_elec_to_CH4_value + costs_CH4_to_demand_value) / CH4_demand_tot  # € / MWh
    lcoe_H2_value = (costs_elec_to_H2_value + costs_H2_to_demand_value) / H2_demand_tot  # € / MWh
    return lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value


def transportation_distribution_cost(model, prediction_transport_and_distrib_annuity):
    """Estimation of annualized transport and distribution cost, based on solar and onshore wind capacities."""
    solar_capacity = value(model.capacity["pv_g"]) + value(model.capacity["pv_c"])
    onshore_capacity = value(model.capacity["onshore"])
    transport_and_distrib_annuity = prediction_transport_and_distrib_annuity["intercept"] + \
                                    prediction_transport_and_distrib_annuity["solar"] * solar_capacity + \
                                    prediction_transport_and_distrib_annuity["onshore"] * onshore_capacity   # 1e9 €/yr
    return transport_and_distrib_annuity

