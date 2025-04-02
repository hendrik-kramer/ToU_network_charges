

import os
import sys
from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt

def model_emob_roll_forsight_smart(timesteplength, spot_prices_xr, network_charges_xr, emob_demand_xr, emob_state_xr, irradiance_xr, parameters):

    
    emob_home_xr = (emob_state_xr=="home")

    dso_names = network_charges_xr.r.to_numpy()

    period_length = 96
    afternoon_offest = 60 # 15 o'clock = 60 = 15 * 4 quarter hours
    departure_time = (9 + 8) * 4   # 9h (15-24) + 8h (until 8am) = 157h * 4 quarters per hour 



    num_periods = int(np.floor(len(spot_prices_xr)/period_length))
    num_regions = len(network_charges_xr.r)
    num_timesteps = period_length * num_periods # passt noch nicht zur xarray-Länge !!!!


    glob_result_p = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])   
    glob_result_p_slack = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])   
    glob_slack_p = xr.DataArray(np.zeros([num_timesteps, num_regions,2]), dims=['t','r','s'])    
    glob_result_E_HStor = xr.DataArray(np.zeros([num_timesteps, num_regions, 2]), dims=['t','r','s']) 

    print(dt.now())

    spot_prices_xr_daily = spot_prices_xr.isel(t=range(0,period_length))
    network_charges_xr_daily = network_charges_xr.isel(t=range(0,period_length))
    emob_demand_xr_daily =  emob_demand_xr.isel(t=range(0,period_length))
    emob_home_xr_daily = emob_home_xr.isel(t=range(0,period_length))
    irradiance_xr_daily = irradiance_xr.isel(t=range(0,period_length))


    # calucalte arrival SOC --> TO BE UPDATED
    e_init_percent = 0.7
    e_max = parameters["ev_soc_max"]
    soc_preference = parameters["ev_soc_preference"]

    e_roll = [] # initialization to prevent "use before assignment" warning below


    # loop over each day
    for ct_roll_horizon in range(1, num_periods):
       
            
        # turn off console completely
        t_subperiod_start = (ct_roll_horizon - 1) * period_length + 1 + afternoon_offest
        print("t_subperiod_start: " + str(t_subperiod_start))
        t_subperiod_end = ct_roll_horizon * period_length + 1 + afternoon_offest
        print("t_subperiod_end: " + str(t_subperiod_end))

        #print(str(t_subperiod_start), " ", str(t_subperiod_end))
    
        # Erstelle ein "Linopy"-Optimierungsmodell
              
        m = Model()
                
        # Definiere Sets
        set_time = pd.Index(range(t_subperiod_start,t_subperiod_end), name="t")
        #print("set_time " , str(list(set_time)))
        set_dso = pd.Index(dso_names, name="r")
        set_vehicle = pd.Index(range(0,len(emob_demand_xr.v)), name="v")
        set_setup = pd.Index(["reg", "red"], name="s")

        
        # =========== VARIABLES ===============
        
        # EV Battery
        SOC_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='SOC_EV', lower=0) # EV battery state of charge
        P_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV', lower=0) # EV charge power
        P_EV_LOSSES = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_EV_LOSSES', lower=0) # EV Battery losses
        P_MOBILITY = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_MOBILITY', lower=0) # EV Mobility
        P_BUY = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_BUY', lower=0) # EV Mobility

        E_BELOW_PREF = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='E_BELOW_PREF', lower=0) # EV Mobility
        P_EV_NOT_HOME = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV_NOT_HOME', lower=0) # EV charge power


        # penalties
        #PENALTY_SOC_EV = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='PENALTY_SOC_EV', lower=0) # EV Mobility


        prosumage = False
        if prosumage == False:
            cons_force_zero1 = m.add_constraints(P_MOBILITY == 0)
            cons_force_zero2 = m.add_constraints(P_EV_LOSSES == 0)


                
        # =========== CONSTRAINTS ===============
        
        # EV Battery
        cons_ev_update = m.add_constraints(SOC_EV.isel(t=range(1,len(set_time))) == SOC_EV.isel(t=range(0,len(set_time)-1)) + timesteplength * (P_EV.isel(t=range(0,len(set_time)-1)) + P_EV_NOT_HOME.isel(t=range(0,len(set_time)-1)) - emob_demand_xr_daily.isel(t=range(0,len(set_time)-1))) , name='cons_storage_update')
        cons_ev_update_last_fix = m.add_constraints(P_EV.isel(t=period_length-1) == 0, name='cons_ev_update_last_fix')
        
        cons_ev_max_soc = m.add_constraints(SOC_EV <= parameters["ev_soc_max"], name='cons_ev_max_soc')
        cons_ev_preference = m.add_constraints(SOC_EV.isel(t=departure_time) >= parameters["ev_soc_preference"] * parameters["ev_soc_max"], name='cons_ev_preference')
        cons_ev_charge_only_home = m.add_constraints(P_EV <= emob_home_xr_daily * parameters["ev_p_ev"], name='cons_ev_charge_only_home')

        # penalize if below preference
        cons_violation_preference  = m.add_constraints(E_BELOW_PREF + SOC_EV >= parameters["ev_soc_preference"] * parameters["ev_soc_max"], name='cons_preference_violation')
        
        
        # PENALTY
        cons_violation_charge_only_home  = m.add_constraints(P_EV_NOT_HOME <= (1-emob_home_xr_daily) * parameters["ev_p_ev"], name='cons_violation_charge_only_home')
        cons_violation_charge_only_home_last_fix = m.add_constraints(P_EV_NOT_HOME.isel(t=period_length-1) == 0, name="cons_violation_charge_only_home_last_fix")


        # Home Balance
        cons_balance = m.add_constraints(P_BUY == P_EV + P_EV_NOT_HOME, name='cons_balance')


        
        #cons_hp_min = m.add_constraints(P_HP - P_HP_slack >= 0, name='power_hp_min')
        #cons_hp_max = m.add_constraints(P_HP - P_HP_slack <= p_hp, name='power_hp_max')
        #cons_stor_max = m.add_constraints(E_HStor <= e_max, name='max_soc')
        #cons_stor_exchange = m.add_constraints(E_HStor.isel(t=range(1,len(set_time)-1)) == E_HStor.isel(t=range(0,len(set_time)-2)) + P_HStor.isel(t=range(0,len(set_time)-2)) * timesteplength, name='system_balance')
        #cons_stor_fix_last = m.add_constraints(P_HStor.isel(t=-1) == 0, name='p_hstor_fix_last_entry')
        # NO CIRCLE IN ROLLING HORIZON!: cons_stor_circle = m.add_constraints(E_HStor.isel(t=0) == E_HStor.isel(t=-2), name='power_hp_circle')
        #cons_demand_balance = m.add_constraints(cop * P_HP - P_HStor - heat_demand_xr == 0, name='demand_balance')
        
        #cons_stor_exchange_max = m.add_constraints(P_HStor <= 1, name='cons_stor_exchange_max')
        #cons_stor_exchange_min = m.add_constraints(P_HStor >= -1, name='cons_stor_exchange_min')
        
              
        # Rolling Constraints
        if ct_roll_horizon > 1:
            cons_roll_init_soc = m.add_constraints(SOC_EV.sel(t=t_subperiod_start) == e_roll, name='cons_rolling_init_soc')
        else:
            cons_roll_init_soc = m.add_constraints(SOC_EV.sel(t=t_subperiod_start) == e_init_percent * e_max, name='cons_rolling_init_soc')
        
        #labels = m.compute_infeasibilities()
        #m.print_infeasibilities()  
        
        # zu minimierende Zielfunktion
        obj = E_BELOW_PREF.sum() + 99999*P_EV_NOT_HOME.sum()

        m.add_objective(obj)
        
        # turn off console off forcefully
    
       
    
        #old_stdout = sys.stdout
        #old_stderr =  sys.stderr 
        #sys.stdout = open(os.devnull, 'w')
        #sys.stderr = open(os.devnull, 'w')
        
        m.solve('gurobi', OutputFlag=1)
        print("=========== day ", str(ct_roll_horizon), " is ", m.status, " ===========") # do not do line breakprint(m.status)
        # turn on console again
        #sys.stdout.close()
        #sys.stderr.close()
        #sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__
        
        labels = m.compute_infeasibilities()
        m.constraints.print_labels(labels)
        
        
        glob_result_p[t_subperiod_start:t_subperiod_end,:,:] = P_HP.solution
        glob_result_p_slack[t_subperiod_start:t_subperiod_end,:,:] = P_HP_slack.solution
        glob_result_E_HStor[t_subperiod_start:t_subperiod_end,:,:] = E_HStor.solution

        e_roll = E_HStor.solution.isel(t=-2)


# do final run

    return m, glob_result_p, glob_result_p_slack, glob_result_E_HStor