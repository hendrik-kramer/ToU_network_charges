

import os
import sys
import warnings
from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt

def model_emob_annual_smart(timesteps, spot_prices_xr, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters, parameters_opti):

    

    quarter = parameters_opti["quarter"]
    time_subset = timesteps[timesteps["Quarter"] == quarter].index #range(0,96*30)
    dso_subset = parameters_opti["dso_subset"]
    emob_subset = parameters_opti["emob_subset"]
    tso_subset = parameters_opti["tso_subset"]
       
    
    timesteps = timesteps.iloc[time_subset]
    spot_prices_xr = spot_prices_xr.isel(t=time_subset)
    network_charges_xr = network_charges_xr.isel(t=time_subset, r=dso_subset)
    emob_demand_xr = emob_demand_xr.isel(t=time_subset, v=emob_subset)
    emob_state_xr = emob_state_xr.isel(t=time_subset, v=emob_subset)
    emob_departure_times = emob_departure_times.iloc[emob_subset]
    
    for key in dict_idx_lookup:
        dict_idx_lookup[key] = dict_idx_lookup[key][dict_idx_lookup[key].isin(time_subset)]

        #dict_idx_lookup[key] = dict_idx_lookup[ct_ev][dict_idx_lookup[ct_ev].isin(time_subset)] for ct_ev in dict_idx_lookup]
    
    
    irradiance_xr = irradiance_xr.isel(t=time_subset, a=tso_subset)

    emob_home_xr = (emob_state_xr=="home")
    emob_HT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").mean()).drop_vars("s")

    dso_names = network_charges_xr.r.to_numpy()

    timesteplength = (timesteps.DateTime.iloc[1] - timesteps.DateTime.iloc[0]).total_seconds()/3600


    print(dt.now())


    # calucalte arrival SOC --> TO BE UPDATED
    e_init_percent = 0.9
    e_max = parameters["ev_soc_max"]
    soc_preference = parameters["ev_soc_preference"]

        


    warnings.simplefilter(action='ignore', category=FutureWarning)  
    warnings.simplefilter(action='ignore', category=UserWarning)      
    
    
    m = Model()
            
    # Definiere Sets
    set_time = pd.Index(timesteps["DateTime"], name="t")
    #print("set_time " , str(list(set_time)))
    set_dso = pd.Index(dso_names, name="r")
    set_region = pd.Index(irradiance_xr.a, name="a")
    set_vehicle = pd.Index(range(0,len(emob_demand_xr.v)), name="v")
    set_setup = pd.Index(["reg", "red"], name="s")

    
    # =========== VARIABLES ===============
    
    #OBJ_WITHOUT_PENALTIES = m.add_variables(name='OBJ_WITHOUT_PENALTIES')
    C_OP = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name='C_OP')

    C_OP_NO_PENALTY = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name="C_OP_NO_PENALTY")

    # EV Battery
    SOC_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='SOC_EV', lower=0) # EV battery state of charge
    P_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV', lower=0) # EV charge power
    P_BUY = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_BUY', lower=0) # EV Mobility
    if parameters_opti["settings_setup"] == "prosumage":
        #BIN_IN = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='BIN_IN', binary=True) # EV Mobility
        P_IN = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_IN', lower=0) # EV Mobility
        P_OUT = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_OUT', lower=0) # EV Mobility
        P_PV = m.add_variables(coords=[set_time, set_dso, set_region, set_setup], name='P_PV', lower=0) # EV Mobility
        SOC_BESS = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='SOC_BESS', lower=0) # EV Mobility




    SOC_BELOW_PREF = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='E_BELOW_PREF', lower=0) # EV Mobility
    P_EV_NOT_HOME = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV_NOT_HOME', lower=0) # EV charge power
    SOC_MISSING = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='SOC_MISSING', lower=0) # EV charge power





            
    # =========== CONSTRAINTS ===============

    # EV Battery
    cons_ev_update = m.add_constraints(SOC_EV.isel(t=range(1,len(set_time))) == SOC_EV.isel(t=range(0,len(set_time)-1)) + timesteplength * (P_EV.isel(t=range(0,len(set_time)-1)) + P_EV_NOT_HOME.isel(t=range(0,len(set_time)-1)) - emob_demand_xr.isel(t=range(0,len(set_time)-1))) , name='cons_ev_update')
    cons_ev_update_last_fix = m.add_constraints(P_EV.isel(t=len(set_time)-1) == 0, name='cons_ev_update_last_fix')
    cons_ev_max_soc = m.add_constraints(SOC_EV <= parameters["ev_soc_max"], name='cons_ev_max_soc')
    cons_ev_charge_only_home = m.add_constraints(P_EV <= emob_home_xr * parameters["ev_p_ev"], name='cons_ev_charge_only_home')

    # dedicated filling level when person usually departs from home
    for ct_ev in emob_subset:
        con_ct_pref_ev = m.add_constraints( SOC_EV.isel(v=ct_ev, t=dict_idx_lookup[emob_departure_times[ct_ev]]) >= - SOC_MISSING.isel(v=ct_ev, t=dict_idx_lookup[emob_departure_times[ct_ev]]) + parameters["ev_soc_departure"] * parameters["ev_soc_max"], name='cons_ct_pref_ev_'+str(ct_ev))
        #del con_ct_pref_ev




    if parameters_opti["settings_setup"] == "prosumage":
         cons_bess_update = m.add_constraints(SOC_BESS.isel(t=range(1,len(set_time))) == SOC_BESS.isel(t=range(0,len(set_time)-1)) + timesteplength * (parameters["stor_eta_in"]*P_IN.isel(t=range(0,len(set_time)-1)) - 1/parameters["stor_eta_out"]*P_OUT.isel(t=range(0,len(set_time)-1))) - parameters["stor_losses"], name='cons_stor_update')
         cons_bess_update_last_fix_in = m.add_constraints(P_IN.isel(t=len(set_time)-1) == 0, name='cons_bess_update_last_fix_in')
         cons_bess_update_last_fix_out = m.add_constraints(P_OUT.isel(t=len(set_time)-1) == 0, name='cons_bess_update_last_fix_out')
 
         cons_bess_max_p_in = m.add_constraints(P_IN <= parameters["stor_p_max"], name='cons_p_max_in')
         cons_bess_max_p_out = m.add_constraints(P_OUT <= parameters["stor_p_max"], name='cons_p_max_out')
         #cons_bess_max_p_bin_in = m.add_constraints(P_IN <= BIN_IN * parameters["stor_p_ev"], name='cons_p_max_in')
         #cons_bess_max_p_bin_out = m.add_constraints(P_OUT <= (-BIN_IN+1) * parameters["stor_p_ev"], name='cons_p_max_out')
         cons_bess_max_soc = m.add_constraints(SOC_BESS <= parameters["stor_soc_max"], name='cons_stor_max_soc')
         cons_bess_circle = m.add_constraints(SOC_BESS.isel(t=1) == SOC_BESS.isel(t=len(set_time)-1))

         cons_pv_p_max = m.add_constraints(P_PV <= (irradiance_xr * parameters["pv_p_max"]), name='cons_pv_p_max')

    if parameters_opti["settings_obj_fnct"] == "scheduled_charging":
        cons_no_HT_charge = m.add_constraints(P_BUY <= emob_HT_xr * 999)      


    # Home Balance
    if parameters_opti["settings_setup"] == "prosumage":
        cons_balance = m.add_constraints(P_BUY + P_EV_NOT_HOME - P_PV + P_OUT - P_IN == P_EV, name='cons_balance')
    else:
        cons_balance = m.add_constraints(P_BUY + P_EV_NOT_HOME == P_EV, name='cons_balance')



    # penalize if below preference
    cons_violation_preference  = m.add_constraints(SOC_EV + SOC_BELOW_PREF >= parameters["ev_soc_preference"] * parameters["ev_soc_max"], name='cons_preference_violation')
    
    
    # PENALTY
    cons_violation_charge_only_home  = m.add_constraints(P_EV_NOT_HOME <= (1-emob_home_xr) * parameters["ev_p_charge_not_home"], name='cons_violation_charge_only_home')
    cons_violation_charge_only_home_last_fix = m.add_constraints(P_EV_NOT_HOME.isel(t=len(set_time)-1) == 0, name="cons_violation_charge_only_home_last_fix")

    cost_xr = np.maximum(network_charges_xr + spot_prices_xr,0)
    cons_cost = m.add_constraints(C_OP ==(cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum(dims="t") , name="cons_cost")

          
    #labels = m.compute_infeasibilities()
    #m.print_infeasibilities()  
    
    # zu minimierende Zielfunktion
    if parameters_opti["settings_obj_fnct"] == "immediate_charging" or parameters_opti["settings_obj_fnct"] == "scheduled_charging":
        obj = 9999 * SOC_BELOW_PREF.sum() + 998*P_EV_NOT_HOME.sum() + 999*SOC_MISSING.sum() 
        cons_obj =  m.add_constraints(C_OP_NO_PENALTY == (cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum(dims="t"))
        #cost_setup_without_penalty = m.add_constraints(C_OP_NO_PENALTY == 999 * SOC_BELOW_PREF.sum())
    
    elif parameters_opti["settings_obj_fnct"] == "smart_charging":    
        obj = (cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum() +  999*SOC_MISSING.sum() 
        cons_obj =  m.add_constraints(C_OP_NO_PENALTY == (cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum(dims="t"))
        
    #elif parameters_opti["settings_obj_fnct"] == "scheduled_charging":    
    #    obj = (cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum() +  999*SOC_MISSING.sum() 
    #    cons_obj =  m.add_constraints(C_OP_NO_PENALTY == (cost_xr * (P_BUY + 2*P_EV_NOT_HOME)).sum(dims="t"))


    m.add_objective(obj)
    
    warnings.simplefilter(action='default', category=FutureWarning)
    warnings.simplefilter(action='default', category=UserWarning)      

    
    return m




    
    #cons_hp_min = m.add_constraints(P_HP - P_HP_slack >= 0, name='power_hp_min')
    #cons_hp_max = m.add_constraints(P_HP - P_HP_slack <= p_hp, name='power_hp_max')
    #cons_stor_max = m.add_constraints(E_HStor <= e_max, name='max_soc')
    #cons_stor_exchange = m.add_constraints(E_HStor.isel(t=range(1,len(set_time)-1)) == E_HStor.isel(t=range(0,len(set_time)-2)) + P_HStor.isel(t=range(0,len(set_time)-2)) * timesteplength, name='system_balance')
    #cons_stor_fix_last = m.add_constraints(P_HStor.isel(t=-1) == 0, name='p_hstor_fix_last_entry')
    # NO CIRCLE IN ROLLING HORIZON!: cons_stor_circle = m.add_constraints(E_HStor.isel(t=0) == E_HStor.isel(t=-2), name='power_hp_circle')
    #cons_demand_balance = m.add_constraints(cop * P_HP - P_HStor - heat_demand_xr == 0, name='demand_balance')
    
    #cons_stor_exchange_max = m.add_constraints(P_HStor <= 1, name='cons_stor_exchange_max')
    #cons_stor_exchange_min = m.add_constraints(P_HStor >= -1, name='cons_stor_exchange_min')
    