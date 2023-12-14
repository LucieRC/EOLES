import pandas as pd
import numpy as np
import logging
import json
import os
import math

def read_hourly_data(config, year):#, scenario='Reference'):#, method="valentin"):#, hourly_heat_elec=None): # calibration=False,
    """Reads data defined at the hourly scale"""
    load_factors = get_pandas(config["load_factors"],
                              lambda x: pd.read_csv(x, index_col=[0, 1], header=None).squeeze("columns"))
    # Get the electricity demand
    demand = get_pandas(config["demand"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW

    # demand_no_residential = process_RTE_demand(config, year, demand, scenario, method=method)#, hourly_residential_heating_RTE=hourly_heat_elec) # calibration=calibration, 

    lake_inflows = get_pandas(config["lake_inflows"],
                              lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh

    data_hourly_and_anticipated = dict()
    data_hourly_and_anticipated["load_factors"] = load_factors
    data_hourly_and_anticipated["demand"] = demand #_no_residential
    data_hourly_and_anticipated["lake_inflows"] = lake_inflows
    return data_hourly_and_anticipated


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




def preprocessing_eoles(anticipated_year_eoles, new_capacity_tot, new_charging_capacity_tot, new_energy_capacity_tot,
                        annualized_costs_new_capacity, annualized_costs_new_energy_capacity, couplingparam,
                        existing_capacity_historical, existing_charging_capacity_historical,
                        existing_energy_capacity_historical, maximum_capacity_evolution, capex_annuity_fOM_historical,
                        storage_annuity_historical, capex_annuity_historical):
    """
    Called for the dynamic versions of EOLES only.
    
    Args:
        new_capacity_tot (): output_dynamics['new_capacity_tot'] 
    """
    #### Get existing and maximum capacities
    existing_capa_historical_y = existing_capacity_historical[[str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
    existing_charging_capacity_historical_y = existing_charging_capacity_historical[[str(anticipated_year_eoles)]].squeeze()
    existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(anticipated_year_eoles)]].squeeze()

    new_maximum_capacity_y = maximum_capacity_evolution[
        [str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

    # Existing capacities at year y
    existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
    # capacities over the whole time horizon before t + existing capacity (from before 2020)
    existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
    existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

    if couplingparam.aggregated_potential:  # we do not take into account previously invested capacity
        maximum_capacity = (existing_capa_historical_y + new_maximum_capacity_y).dropna()
    else:
        maximum_capacity = (
                existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to
        # technologies without any upper bound

    #### Historical annualized costs based on historical costs
    annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
        existing_capa_historical_y, capex_annuity_fOM_historical, existing_energy_capacity_historical_y,
        storage_annuity_historical)

    annualized_costs_capacity_nofOM_historical = eoles.utils.annualized_costs_investment_historical_nofOM(
        existing_capa_historical_y, capex_annuity_historical, existing_energy_capacity_historical_y,
        storage_annuity_historical)

    ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
    # Necessary for calculus of LCOE accounting for evolution of capacities
    annualized_costs_capacity = pd.concat(
        [annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
         annualized_costs_new_capacity], axis=1)
    annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + \
                                                    annualized_costs_capacity['annualized_costs']
    annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
        columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
    annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity[
                                                               'historical_annualized_costs'] + \
                                                           annualized_costs_energy_capacity['annualized_costs']

    preprocessing = {
        'existing_capacity': existing_capacity,
        'existing_charging_capacity': existing_charging_capacity,
        'existing_energy_capacity': existing_energy_capacity,
        'maximum_capacity': maximum_capacity,
        'annualized_costs_capacity': annualized_costs_capacity,
        'annualized_costs_energy_capacity': annualized_costs_energy_capacity,
        'annualized_costs_capacity_nofOM_historical': annualized_costs_capacity_nofOM_historical,
        'annualized_costs_energy_capacity_historical': annualized_costs_energy_capacity_historical
    }
    return preprocessing