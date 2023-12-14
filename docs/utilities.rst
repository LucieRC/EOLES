======================
Utilities
======================

Introduction
------------

This page provides detailed documentation for the "utils.py" file, \
which includes a variety of utility functions and classes used for data processing, \
plotting, and optimization model management.

Dependencies
------------

- `dataclasses`: Used for creating data classes in Python.
- `math`: Provides access to mathematical functions.
- `importlib`: Enables the import of resources within the package.
- `pathlib`: Used for handling file paths.
- `pandas (pd)`: For data manipulation and analysis.
- `numpy (np)`: Supports large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.
- `matplotlib.pyplot (plt)`: For creating static, interactive, and animated visualizations in Python.
- `pyomo.environ`: Provides an environment for defining optimization models.
- `datetime`: For manipulating dates and times.
- `copy`: Used for shallow and deep copy operations.
- `typing`: Provides runtime support for type hints.
- `pickle`: For serializing and de-serializing Python object structures.
- Others: Additional libraries as used in the file.

Utility Functions
------------------

`get_pandas`
==========================

Function used to read input data. It likely involves reading and processing data into pandas DataFrame format.

`get_config`
==========================

This function does not have a docstring, so its purpose is unclear. It might be related to configuration settings or parameters.

`process_heating_need`
==========================

Transforms the index of heating need into the number of hours. It takes a pandas DataFrame with hourly heating need and a climate year as parameters.

`calculate_hp_cop`
==========================

Calculates the heat pump coefficient based on renewable ninja data. This function likely computes the coefficient of performance (COP) for heat pumps.

`heating_hourly_profile`
==========================
Creates an hourly profile. The exact nature of this profile is not specified in the docstring, but it could be related to heating demand or energy consumption.

`load_evolution_data`
==========================
The ``load_evolution_data`` function is designed to load essential data required for the social planner trajectory in energy systems modeling. It performs the following key operations:

1. **Load Historical Data**:
   - ``existing_capacity_historical``: This loads historical data on existing capacity (measured in GW) from a specified CSV file, representing historical figures for various energy technologies or resources.
   - ``existing_charging_capacity_historical``: It loads historical data on charging capacities (in GW) from a CSV file, which might relate to energy storage systems or similar technologies.
   - ``existing_energy_capacity_historical``: Imports historical data on energy capacities (in GW) from a CSV file, likely related to energy storage or generation capacities.
2. **Load Maximum Capacity Evolution Data**:
   - ``maximum_capacity_evolution``: Retrieves data concerning the evolution of maximum capacity (in GW), as specified in the provided `config`. This data is filtered by a particular scenario indicated in the `config`.
3. **Load Historical Cost Data**:
   - ``capex_annuity_fOM_historical``: Loads historical data on capital expenditure annuity and fixed operation and maintenance costs from a CSV file.
   - ``capex_annuity_historical``: Imports data related to historical capital expenditure annuities from a CSV file.
   - ``storage_annuity_historical``: Retrieves historical data concerning storage annuities from a CSV file.
4. **Import Evolution of Tertiary and ECS Gas Demand**:
   - ``heating_gas_demand_RTE_timesteps``: This loads data on the evolution of heating gas demand for tertiary applications, segmented into specific time steps.
   - ``ECS_gas_demand_RTE_timesteps``: Imports data on the evolution of gas demand for Energy Conservation Systems (ECS), also segmented into specific time steps.

All the loaded data are graphed for visualization in :ref:`input_data`.
The function returns a tuple containing all these datasets, playing a crucial role in providing historical and scenario-based data essential for planning and analyzing social planner trajectories in energy system models.

`process_RTE_demand`
==========================

`profile_ev`
==========================


`calculate_annuities_capex`
============================


`calculate_annuities_storage_capex`
=====================================


`calculate_annuities_renovation`
==================================


`calculate_annuities_resirf`
================================


`update_ngas_cost`
==========================


`create_hourly_residential_demand_profile`
============================================


`define_month_hours`
==========================


`get_technical_cost`
==========================


`extract_capacities`
==========================

...

`annualized_costs_investment_historical`
==========================================

`annualized_costs_investment_historical_nofOM`
===============================================


Usage Examples
---------------

.. note:: Provide usage examples for some of the key functions.

.. code-block:: python

    # Example usage of function_name_1
    result = function_name_1(param1, param2)

    # Example usage of function_name_2
    result = function_name_2(param1)

Conclusion
----------

For specific details about each function, please refer to the function definitions within the file.
