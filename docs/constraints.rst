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
    \sum_{h=8760y}^{8760(y+1)-1} gene_{nuc, h} \leq cf_{nuc} \cdot capacity_{nuc} \cdot 8760 \quad \forall y \in Y

Hourly Nuclear Production Constraint
-------------------------------------
.. math::
    capacity_{nuc} \cdot cf\_nuc\_hly \geq gene_{nuc, h} \quad \forall h \in H

Maximum Power Constraint for Non-VRE Technologies
-------------------------------------------------
.. math::
    capacity_{tec} \geq gene_{tec, h} \quad \forall h \in H, tec \in TEC

Battery 1h Capacity Constraint
------------------------------
.. math::
    capacity_{battery1} = energy\_capacity_{battery1}

Battery 4h Capacity Constraint
------------------------------
.. math::
    capacity_{battery4} = \frac{energy\_capacity_{battery4}}{4}

FRR Capacity Constraint
-----------------------
.. math::
    capacity_{frr} \geq gene_{frr, h} + reserve_{frr, h} \quad \forall frr \in FRR, h \in H

Energy Storage Consistency Constraint
-------------------------------------
.. math::
    \begin{equation}
    \begin{aligned}
        \text{if } h < (last\_h - 1): \quad & stored_{storage\_tecs, h+1} = stored_{storage\_tecs, h} 
        + \left( storage_{str\_tec, h} \cdot \eta^{in}_{str\_tec} \right) 
        - \left( \frac{gene_{str\_tec, h}}{\eta^{out}_{storage\_tecs}} \right) \\
        \text{else}: \quad & stored_{str\_tec, 0} = stored_{str\_tec, h} 
        + \left( storage_{str\_tec, h} \cdot \eta^{in}_{str\_tec} \right) 
        - \left( \frac{gene_{str\_tec, h}}{\eta^{out}_{str\_tec}} \right) \\
    \end{aligned}
    \end{equation}

First and Last Hour Storage Equality Constraint
-----------------------------------------------
.. math::


Maximum Monthly Lake Generation Constraint
-------------------------------------------
.. math::


Maximum Stored Energy Constraint
--------------------------------
.. math::


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
