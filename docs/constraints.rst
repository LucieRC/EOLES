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
    stored_{str\_tec, first\_h} = stored_{str\_tec, last\_h - 1} + \left( storage_{str\_tec, last\_h - 1} \cdot \eta^{in}_{str\_tec} \right) - \left( \frac{gene_{str\_tec, last\_h - 1}}{\eta^{out}_{str\_tec}} \right)


Maximum Monthly Lake Generation Constraint
-------------------------------------------
.. math::
    \sum_{{h} \in months\_hs_{m}} gene_{lake, h} \leq lake\_inflows_{m} \cdot 1000 \quad \forall m \in MONTHS

Maximum Stored Energy Constraint
--------------------------------
.. math::
    stored_{str\_tec, h} \leq energy\_capacity_{str\_tec} \quad \forall str\_tec \in STR\_TEC, h \in H

Storage Charging Capacity Constraint
------------------------------------
.. math::
    storage_{str\_tec, h} \leq charging\_capacity_{str\_tec} \quad \forall str\_tec \in STR\_TEC, h \in H

Hydrogen Discharge Capacity Constraint
--------------------------------------
.. math::
    capacity_{H2} \leq \frac{energy\_capacity_{H2} \cdot 26}{3000}

Hydrogen Charging Capacity Constraint
-------------------------------------
.. math::
    charging\_capacity_{H2} \leq \frac{energy\_capacity_{H2} \cdot 6.4}{3000}

PHS Charging Capacity Constraint
--------------------------------
.. math::
    charging\_capacity_{phs} \leq capacity_{phs} \cdot 0.91

Battery Capacity Equality Constraint
------------------------------------
.. math::
    charging\_capacity_{battery} = capacity_{battery} \quad \forall battery in BATTERY

Storage Charging and Discharging Capacity Constraint
----------------------------------------------------
.. math::
    charging\_capacity_{str\_tec} \leq capacity_{str\_tec}

Annual Methanization Power Production Constraint
------------------------------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{meth, h} \leq biomass\_potential_{methani} \cdot 1000 \quad \forall y \in Y

Annual Pyrogazification Power Production Constraint
---------------------------------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{pyro, h} \leq biomass\_potential_{pyr} \cdot 1000 \quad \forall y \in Y

Geothermal Potential Constraint
-------------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{geo, h} \leq district\_heating\_potential_{geoth} \cdot 1000 \quad \forall y \in Y

Central Wood Boiler Potential Constraint
----------------------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{central_wood, h} \leq district\_heating\_potential_{central\_wood\_boiler} \cdot 1000 \quad \forall y \in Y

UIOM Potential Constraint
-------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{UIOM, h} \leq district\_heating\_potential_{UIOM} \cdot 1000 \quad \forall y \in Y

FRR Reserves Constraint
-----------------------
.. math::
    \sum_{frr} reserve_{frr, h} = \sum_{vre} \epsilon_{vre} \cdot capacity_{vre} + elec\_demand_{h} \cdot load\_uncertainty \cdot (1 + \delta) \quad \forall h \in H

Hydrogen Balance Constraint
---------------------------
.. math::
    gene_{electro, h} + gene_{H2, h} = \frac{gene_{H2\_CCGT, h}}{conv\_{\eta}_{H2\_CCGT}} + H2\_demand_{h} + storage_{H2, h} \quad \forall h \in H

Methane Balance Constraint
--------------------------
.. math::
    gene_{methana, h} + gene_{methani, h} + gene_{pyr, h} + gene_{CH4, h} + gene_{nat\_gas, h} = \\
     \quad \frac{gene_{OCGT, h}}{conv\_{\eta}_{OCGT}} + \frac{gene_{CCGT, h}}{conv\_{\eta}_{CCGT}} + gene_{central\_gas\_boiler, h} + CH4\_demand_{h} + storage_{CH4, h} \quad \forall h \in H

District Heating Balance Constraint
-----------------------------------
.. math::
    \sum_{tec} gene_{tec, h} \cdot conv\_{\eta}_{tec} \geq district\_heating\_demand_{h} \quad \forall h \in H, tec \in DH\_balance

Electricity Adequacy Constraint
--------------------------------
.. math::
    \sum_{str\_elec} storage_{str\_elec, h} + \frac{gene_{electro, h}}{conv\_{\eta}_{electro}} + \frac{gene_{methana, h}}{conv\_{\eta}_{methana}} + \sum_{elec\_balance} gene_{elec\_balance, h} \geq elec\_demand_{h} \quad \forall h \in H

Nuclear Ramping Up Flexibility Constraint
-----------------------------------------
.. math::
    gene_{nuc, h} - gene_{nuc, h-1} + reserve_{nuc, h} - reserve_{nuc, h-1} \leq hourly\_ramp\_nuc \cdot capacity_{nuc} \quad \forall h \in H

Nuclear Ramping Down Flexibility Constraint
--------------------------------------------
.. math::
    gene_{nuc, h-1} - gene_{nuc, h} + reserve_{nuc, h-1} - reserve_{nuc, h} \leq hourly\_ramp\_nuc \cdot capacity_{nuc} \quad \forall h \in H

Methanation CO2 Balance Constraint
----------------------------------
.. math::
    \frac{\sum_{h=8760y}^{8760(y+1)-1} gene_{methana, h}}{conv\_{\eta}_{methana}} \leq \sum_{h=8760y}^{8760(y+1)-1} gene_{methani, h} \cdot \%\_CO2\_from\_methani \quad \forall y \in Y

Carbon Budget Constraint
-------------------------
.. math::
    \sum_{h=8760y}^{8760(y+1)-1} gene_{nat\_gas, h} \cdot \frac{0.2295}{1000} \leq carbon\_budget \quad \forall y \in Y
