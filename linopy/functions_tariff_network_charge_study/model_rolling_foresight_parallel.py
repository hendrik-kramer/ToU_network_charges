

import os
import sys
from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt

def build_hp_model(prices, dsos, prices_xr, e_max, e_init_percent, e_min_end_percent, p_hp, cop, heat_demand_xr, penalty, timesteplength):


    period_length = 96
    num_periods = int(np.floor(len(prices_xr)/period_length))

    num_regions = len(dsos)
    num_timesteps = period_length * num_periods # passt noch nicht zur xarray-Länge !!!!


    glob_result_p = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])   
    glob_result_p_slack = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])   
    glob_slack_p = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])    
    glob_result_E_HStor = xr.DataArray(np.zeros([num_timesteps, num_regions, 2]), dims=['t','r','s']) 

    print(dt.now())

    # loop over each day
    for ct_roll_horizon in range(1, num_periods):
        
            
        # turn off console completely
        t_subperiod_start = (ct_roll_horizon - 1) * period_length + 1
        t_subperiod_end = ct_roll_horizon * period_length + 1

        #print(str(t_subperiod_start), " ", str(t_subperiod_end))
    
        # Erstelle ein "Linopy"-Optimierungsmodell
              
        m = Model()
                
        # Definiere Sets
        set_time = pd.Index(range(t_subperiod_start,t_subperiod_end), name="t")
        #print("set_time " , str(list(set_time)))
        set_dso = pd.Index(dsos, name="r")
        set_setup = pd.Index(["reg", "red"], name="s")
        
        
        # Endogeneous Variablen
        P_HP = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP', lower=0) # Leistung Wärmepumpe
        P_HP_slack = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP_slack', lower=0) # Leistung Wärmepumpe
        
        P_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HStor') # EINspeicherung SPeicher
        E_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='E_HStor', lower=0) # EINspeicherung SPeicher
        
        # Constraints
        cons_hp_min = m.add_constraints(P_HP - P_HP_slack >= 0, name='power_hp_min')
        cons_hp_max = m.add_constraints(P_HP - P_HP_slack <= p_hp, name='power_hp_max')
        cons_stor_max = m.add_constraints(E_HStor <= e_max, name='max_soc')
        cons_stor_exchange = m.add_constraints(E_HStor.isel(t=range(1,len(set_time)-1)) == E_HStor.isel(t=range(0,len(set_time)-2)) + P_HStor.isel(t=range(0,len(set_time)-2)) * timesteplength, name='system_balance')
        cons_stor_fix_last = m.add_constraints(P_HStor.isel(t=-1) == 0, name='p_hstor_fix_last_entry')
        # NO CIRCLE IN ROLLING HORIZON!: cons_stor_circle = m.add_constraints(E_HStor.isel(t=0) == E_HStor.isel(t=-2), name='power_hp_circle')
        cons_demand_balance = m.add_constraints(cop * P_HP - P_HStor - heat_demand_xr == 0, name='demand_balance')
        
        cons_stor_exchange_max = m.add_constraints(P_HStor <= 1, name='cons_stor_exchange_max')
        cons_stor_exchange_min = m.add_constraints(P_HStor >= -1, name='cons_stor_exchange_min')
        
              
        # Rolling Constraints
        if ct_roll_horizon > 1:
            cons_roll_init_soc = m.add_constraints(E_HStor.sel(t=t_subperiod_start) == e_roll, name='cons_rolling_init_soc')
        else:
            cons_roll_init_soc = m.add_constraints(E_HStor.sel(t=t_subperiod_start) == e_init_percent * e_max, name='cons_rolling_init_soc')
        
        cons_roll_end_soc = m.add_constraints(E_HStor.sel(t=t_subperiod_end-2) >= e_min_end_percent * e_max, name='cons_rolling_end_soc')

        #labels = m.compute_infeasibilities()
        #m.print_infeasibilities()  
        
        # zu minimierende Zielfunktion
        obj = (prices_xr * P_HP).sum() + (penalty * P_HP_slack).sum()
        m.add_objective(obj)
        
        # turn off console off forcefully
    
        #old_stdout = sys.stdout
        #old_stderr =  sys.stderr 
        #sys.stdout = open(os.devnull, 'w')
        #sys.stderr = open(os.devnull, 'w')
        
        m.solve('gurobi', OutputFlag=0)
        print("=========== day ", str(ct_roll_horizon), " is ", m.status, " ===========") # do not do line breakprint(m.status)
        # turn on console again
        #sys.stdout.close()
        #sys.stderr.close()
        #sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__
        
        
        glob_result_p[t_subperiod_start:t_subperiod_end,:,:] = P_HP.solution
        glob_result_p_slack[t_subperiod_start:t_subperiod_end,:,:] = P_HP_slack.solution
        glob_result_E_HStor[t_subperiod_start:t_subperiod_end,:,:] = E_HStor.solution

        e_roll = E_HStor.solution.isel(t=-2)


# do final run

    return m, glob_result_p, glob_result_p_slack, glob_result_E_HStor