##########################################
Objective function
##########################################

.. math::
  \begin{align*}
  	& (  {\sum_{{t \in T}}} (capa_{t} - exist\_capa_{t}) \cdot a_{t} \cdot nb\_y \\ 
  	& + \sum_{{t \in T}} capa_{t} \cdot fOM_t \cdot nb\_y \\
  	& + \sum_{{s \in S}} (NRJ\_capa_{s} - exist\_NRJ\_capa_{s}) \cdot str\_a_{s} \cdot nb\_y \\ 
  	& + \sum_{{s \in S}} (ch\_capa_{s} - exist\_ch\_capa_{s}) \cdot ch\_capex_{s} \cdot nb\_y \\  
  	& + \sum_{{s \in S}} ch\_capa_{s} \cdot ch\_opex_{s} \cdot nb\_y \\ 
  	& + \sum_{{t \in T}} \sum_{{h \in H}} gene_{t, h} \cdot vOM_{t} ) /1000
  \end{align*}
