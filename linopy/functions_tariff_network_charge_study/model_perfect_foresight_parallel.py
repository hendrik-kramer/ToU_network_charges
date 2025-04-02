from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd

def build_hp_model(prices, dsos, prices_xr, e_max, p_hp, cop, heat_demand_xr, penalty, timesteplength):


    # Erstelle ein "Linopy"-Optimierungsmodell
    m = Model()
    
    
    # Definiere Sets
    set_time = pd.Index(range(0,len(prices)), name="t")
    set_dso = pd.Index(dsos, name="r")
    set_setup = pd.Index(["reg", "red"], name="s")
    
    
    # Endogene Variablen
    P_HP = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP', lower=0) # Leistung Wärmepumpe
    P_HP_slack = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP_slack', lower=0) # Leistung Wärmepumpe
    
    P_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HStor') # EINspeicherung SPeicher
    E_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='E_HStor', lower=0) # EINspeicherung SPeicher
    
    # Nebenbedingugen
    cons_hp_min = m.add_constraints(P_HP - P_HP_slack >= 0, name='power_hp_min')
    cons_hp_max = m.add_constraints(P_HP - P_HP_slack <= p_hp, name='power_hp_max')
    cons_stor_max = m.add_constraints(E_HStor <= e_max, name='max_soc')
    cons_stor_exchange = m.add_constraints(E_HStor.isel(t=range(1,len(set_time)-1)) == E_HStor.isel(t=range(0,len(set_time)-2)) + P_HStor.isel(t=range(0,len(set_time)-2)) * timesteplength, name='system_balance')
    cons_stor_fix_last = m.add_constraints(P_HStor.isel(t=-1) == 0, name='p_hstor_fix_last_entry')
    cons_stor_circle = m.add_constraints(E_HStor.isel(t=0) == E_HStor.isel(t=-2), name='power_hp_circle')
    cons_demand_balance = m.add_constraints(cop * P_HP - P_HStor - heat_demand_xr == 0, name='demand_balance')
    
    cons_stor_exchange_max = m.add_constraints(P_HStor <= 1, name='cons_stor_exchange_max')
    cons_stor_exchange_min = m.add_constraints(P_HStor >= -1, name='cons_stor_exchange_min')
    
    
    
    # zu minimierende Zielfunktion
    obj = (prices_xr * P_HP).sum() + (penalty * P_HP_slack).sum()
    m.add_objective(obj)

    m.solve('gurobi')
    
    print(m.solution[1])




    return m

