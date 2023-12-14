##########################################
Constraints
##########################################

Part under development.

Generation Constraint for VRE Profiles
---------------------------------------

.. math::
   gene_{vre, h} = capacity_{vre} \cdot lf_{vre, h} \quad \forall h \in H, vre \in VRE


Annual Nuclear Production Constraint
------------------------------------

.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{nuc, h} \leq cf_{nuc} \cdot capacity_{nuc} \cdot 8760 && \forall y \in Y

Hourly Nuclear Production Constraint
-------------------------------------

Maximum Power Constraint for Non-VRE Technologies
-------------------------------------------------

Battery 1h Capacity Constraint
------------------------------

Battery 4h Capacity Constraint
------------------------------

FRR Capacity Constraint
-----------------------

Energy Storage Consistency Constraint
-------------------------------------

First and Last Hour Storage Equality Constraint
-----------------------------------------------

Maximum Monthly Lake Generation Constraint
-------------------------------------------

Maximum Stored Energy Constraint
--------------------------------

Storage Charging Capacity Constraint
------------------------------------

Hydrogen Discharge Capacity Constraint
--------------------------------------

Hydrogen Charging Capacity Constraint
-------------------------------------

PHS Charging Capacity Constraint
--------------------------------

Battery Capacity Equality Constraint
------------------------------------

Storage Charging and Discharging Capacity Constraint
----------------------------------------------------

Annual Methanization Power Production Constraint
------------------------------------------------

Annual Pyrogazification Power Production Constraint
---------------------------------------------------

Geothermal Potential Constraint
-------------------------------

Central Wood Boiler Potential Constraint
----------------------------------------

UIOM Potential Constraint
-------------------------

FRR Reserves Constraint
-----------------------

Hydrogen Balance Constraint
---------------------------

Methane Balance Constraint
--------------------------

District Heating Balance Constraint
-----------------------------------

Electricity Adequacy Constraint
--------------------------------

Nuclear Ramping Up Flexibility Constraint
-----------------------------------------

Nuclear Ramping Down Flexibility Constraint
--------------------------------------------

Methanation CO2 Balance Constraint
----------------------------------

Carbon Budget Constraint
-------------------------
