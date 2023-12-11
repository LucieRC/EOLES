"""
Power system components.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import math
from eoles.utils import get_pandas, process_RTE_demand, calculate_annuities_capex, calculate_annuities_storage_capex, \
    update_ngas_cost, define_month_hours, calculate_annuities_renovation, get_technical_cost, extract_hourly_generation, \
    extract_spot_price, extract_capacities, extract_energy_capacity, extract_supply_elec, extract_primary_gene, \
    extract_use_elec, extract_renovation_rates, extract_heat_gene, calculate_LCOE_gene_tec, calculate_LCOE_conv_tec, \
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


# file_handler = logging.FileHandler('root_log.log')
# file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
# logger.addHandler(file_handler)


class ModelEOLES():
    def __init__(self, name, config, path, logger, hourly_heat_elec, hourly_heat_gas, hourly_heat_district=None,
                 wood_consumption=0, oil_consumption=0,
                 existing_capacity=None, existing_charging_capacity=None, existing_energy_capacity=None, maximum_capacity=None,
                 method_hourly_profile="valentin", anticipated_social_cost_of_carbon=0, actual_social_cost_of_carbon=0, year=2050, anticipated_year=2050,
                 scenario_cost=None, existing_annualized_costs_elec=0,
                 existing_annualized_costs_CH4=0, existing_annualized_costs_H2=0, existing_annualized_costs_CH4_naturalgas=0,
                 existing_annualized_costs_CH4_biogas=0, carbon_constraint=False, discount_rate=0.045, calibration=False):
        """

        :param name: str
        :param config: dict
        :param path: str
        :param logger:
        :param hourly_heat_elec: pd.Series
            Sequence of hourly electricity demand for heat in the residential sector. 
        :param hourly_heat_gas: pd.Series
            Sequence of hourly gas demand for heat in the residential sector
        :param wood_consumption: float
        :param oil_consumption: float
        :param existing_capacity: pd.Series
        :param existing_charging_capacity: pd.Series
        :param existing_energy_capacity: pd.Series
        :param maximum_capacity: pd.Series
        :param method_hourly_profile: str
            Method to calculate the hourly profile for electricity and gas demand related to heat
        :param anticipated_social_cost_of_carbon: int
            Anticipated social cost of carbon used to calculate emissions and to find optimal power mix.
        :param actual_social_cost_of_carbon
            Actual social cost of carbon, used when calculating the real functionment cost in post processing.
        :param year: int
        :param anticipated_year: int
        :param scenario_cost: dict
        :param existing_annualized_costs_elec: float
        :param existing_annualized_costs_CH4: float
        :param existing_annualized_costs_CH4_naturalgas: float
            Existing costs related to natural gas + methane
        :param existing_annualized_costs_CH4: float
            Existing costs related to biogas (methanization, pyro, methanation)
        :param existing_annualized_costs_H2: float
        :param carbon_constraint: bool
            If true, include a carbon constraint instead of the social cost of carbon
        :param discount_rate: float
            Discount rate used to calculate annuities
        """
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

        assert hourly_heat_elec is not None, "Hourly electricity heat profile should be provided to the model"
        assert hourly_heat_gas is not None, "Hourly gas heat profile should be provided to the model"

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
        data_technology = read_technology_data(self.config, self.year)      # get current technology data
        data_annual = read_annual_data(self.config, self.anticipated_year)  # get anticipated demand and energy prices
        data_technology.update(data_annual)
        data_static = data_technology
        if scenario_cost is not None:                                       # we update costs based on data given in scenario
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
        self.model.h = RangeSet(self.first_hour, self.last_hour - 1)
        # Months
        self.model.months = RangeSet(1, 12 * self.nb_years)
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
            """Returns the existing capacity and its maximum if defined"""
            if i in self.maximum_capacity.keys():  # there exists a max capacity
                return self.existing_capacity[i], self.maximum_capacity[i]  # existing capacity is always the lower bound
            else:
                return self.existing_capacity[i], None  # in this case, only lower bound exists

        def charging_capacity_bounds(model, i):
            """Does nothing"""
            # TODO: j'ai enlevé cette contrainte, car je suppose ici que la seule contrainte provient de la discharging capacity
            # if i in self.maximum_charging_capacity.keys():
            #     return self.existing_charging_capacity[i], self.maximum_capacity[i]
            # else:
            return self.existing_charging_capacity[i], None

        def energy_capacity_bounds(model, i):
            """Returns the max energy capacity and its maximum if defined"""
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
            if tec in self.fix_charging_capacities.keys():
                self.model.charging_capacity[tec].fix(self.fix_charging_capacities[tec])
            if tec in self.fix_energy_capacities.keys():
                self.model.energy_capacity[tec].fix(self.fix_energy_capacities[tec])

    def define_constraints(self):
        def generation_vre_constraint_rule(model, h, vre):
            """Constraint on variables renewable profiles generation."""
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


