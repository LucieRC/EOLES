.. EOLES documentation master file, created by
   sphinx-quickstart on Sun Dec 10 13:49:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################################################################
EOLES - Energy Optimization for Low Emission Systems
##################################################################################

The EOLES model optimizes the investment and operation of the energy system in order to minimize its total cost (possibly including a cost of CO2 emissions) while satisfying energy demand. The first model was developed at CIRED by Behrang Shirizadeh, Quentin Perrier and Philippe Quirion.

Here are a few questions these models can address:

- What is the optimal energy mix in the long run, taking into account CO2 emissions?
- If a particular technology turns out to be more expensive than expected, or if it cannot be used for whatever reason, how does it impact the energy mix, the energy system cost and CO2 emissions?
- To what extent are the energy system cost, the electricity price and the optimal energy mix sensitive to the electricity or hydrogen demand?
- How does sector coupling, and more generally the coupling  between different energy carriers (electricity, hydrogen, methane, heat networks…) change the optimal energy mix?
- What are the conditions needed to reach carbon-neutrality in the energy system?


.. toctree::
   :maxdepth: 1
   :caption: Model components

   sets
   variables
   constraints
   objective_function
   
.. toctree::
   :maxdepth: 1
   :caption: Model description

   introduction
   utilities