import os
import sys
import warnings
from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt
import random

def model_emob_quarter_smart2(timesteps, cost_home_xr, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, parameters_model, parameters_opti):

    timesteplength = (timesteps.DateTime.iloc[1] - timesteps.DateTime.iloc[0]).total_seconds()/3600

    # calucalte arrival SOC --> TO BE UPDATED
    soc_preference = parameters_model["ev_soc_preference"]
    e_ev_init_percent = parameters_model["ev_soc_init_rel"]
    cost_public = parameters_model["cost_public_charge_pole"]  # ct/kWh
    
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    warnings.simplefilter(action='ignore', category=UserWarning)      


    emob_home_xr = (emob_state_xr=="home")
    emob_public_xr = (emob_state_xr=="workplace") | (emob_state_xr=="errands") | (emob_state_xr=="escort") | (emob_state_xr=="leisure") | (emob_state_xr=="shopping")
    emob_HT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").mean(dim="t")).drop_vars("s")
    emob_STHT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").min("t") + 0.01).drop_vars("s")

    emob_true_when_geting_home_xr = emob_home_xr & (~emob_home_xr.shift(t=1, fill_value=False))
    emob_first_id_when_getting_home_xr =  emob_true_when_geting_home_xr.argmax('t').where(emob_true_when_geting_home_xr.any('t'), other=np.nan)
    time_idx = xr.DataArray(np.arange(emob_home_xr.sizes['t']), dims='t', coords={'t': emob_home_xr['t']})
    emob_true_before_geting_home_xr = (time_idx < emob_first_id_when_getting_home_xr)
    
    # add true values if there is not trip that day to assume no charging before typical hour of return of this vehicle pattern 
    id_no_trip_today = (emob_true_before_geting_home_xr.isel(t=0) == False)
  
    for idx in np.where(id_no_trip_today.values)[0]:  
        random_value = np.random.randint(1, 6*4) # first 6h with 4 *15min
        emob_true_before_geting_home_xr[:, idx][:random_value] = True


    m = Model()
            
    # Definiere Sets
    set_time = pd.Index(timesteps["DateTime"], name="t")
    set_dso = pd.Index(network_charges_xr["r"].values, name="r")
    set_vehicle = pd.Index(emob_demand_xr["v"].values, name="v")
    set_setup = pd.Index(["reg", "red"], name="s")


    # =========== VARIABLES ===============

    # cost per timestep (prices x charge power)
    C_ALL_planning = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name='C_ALL_planning', lower=0) # including overlapping period
    C_HOME_planning = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name="C_HOME_planning", lower=0)
    C_PUBLIC_planning = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name='C_PUBLIC_planning', lower=0) # including overlapping period

    C_ALL = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name='C_ALL', lower=0) # excluding overlapping period
    C_HOME = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name="C_HOME", lower=0)
    C_PUBLIC = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name="C_PUBLIC", lower=0)


    # Charge power
    P_HOME = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_HOME', lower=0) # EV Mobility
    P_PUBLIC = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_PUBLIC', lower=0) # EV Mobility


    # EV Battery
    P_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV', lower=0) # EV charge power
    SOC_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='SOC_EV', lower=0) # EV battery state of charge

    PENALTY_TIMEPREF  = m.add_variables(coords=[set_dso,set_vehicle, set_setup], name='PENALTY_TIMEPREF', lower=0) # discomfort when charging during ST HT segment, only  relevant for red network charge
    PENALTY_ST_HT  = m.add_variables(coords=[set_dso,set_vehicle, set_setup], name='PENALTY_ST_HT', lower=0) # discomfort when charging during ST HT segment, only  relevant for red network charge
    PENALTY_NO_CHARGE_BEFORE_ARRIVAL = m.add_variables(coords=[set_dso,set_vehicle, set_setup], name='PENALTY_NO_CHARGE_BEFORE_ARRIVAL', lower=0) 
    



    # =========== CONSTRAINTS COST ===============
   

    # cost of whole planning horizon for objective function --> sum(dims="t")
    cons_cost_home_planning = m.add_constraints(C_HOME_planning     == timesteplength * (cost_home_xr * P_HOME).sum(dims="t"), name="cons_cost_home_whole_planning_period")
    cons_cost_public_planning = m.add_constraints(C_PUBLIC_planning == timesteplength * (cost_public * P_PUBLIC).sum(dims="t"), name="cons_cost_public_whole_planning_period")
    cons_cost_all_planning = m.add_constraints(C_ALL_planning       == C_HOME_planning + C_PUBLIC_planning, name="cons_cost_all_whole_planning_period")


    # filter non-overlapping hours to save cost sum --> t=idx_save
    idx_save = list(timesteps[timesteps.save_day_data].counter_id)
    cons_cost_home_1pm = m.add_constraints(C_HOME       == timesteplength * (cost_home_xr * P_HOME).isel(t=idx_save).sum(dims="t"), name="cons_cost_home_1pm")
    cons_cost_public_1pm = m.add_constraints(C_PUBLIC   == timesteplength * (cost_public * P_PUBLIC).isel(t=idx_save).sum(dims="t"), name="cons_cost_public_1pm")
    cons_cost_all_1pm = m.add_constraints(C_ALL         == C_HOME + C_PUBLIC , name="cons_cost_all_1pm")

    



        
    # =========== CONSTRAINTS ===============


    # EV Battery behavior
    cons_ev_update = m.add_constraints(SOC_EV.isel(t=range(1,len(set_time))) == SOC_EV.isel(t=range(0,len(set_time)-1)) 
                                                                       + timesteplength * (parameters_model["ev_eta_in"] * P_EV.isel(t=range(0,len(set_time)-1)) - 1/parameters_model["ev_eta_in"]*emob_demand_xr.isel(t=range(0,len(set_time)-1)) - parameters_model["ev_losses"] ) , name='cons_ev_update')
    
    cons_ev_max_soc = m.add_constraints(SOC_EV <= parameters_model["ev_soc_max"], name='cons_ev_max_soc')
    

    # dedicated filling level when person usually departs from home
    for ct_ev in set_vehicle.astype(int):
        t_ev = dict_idx_lookup[emob_departure_times[ct_ev]]
        con_ct_pref_ev = m.add_constraints( SOC_EV.isel(v=ct_ev, t=dict_idx_lookup[emob_departure_times[ct_ev]]) >= parameters_model["ev_soc_departure"] * parameters_model["ev_soc_max"], name='cons_ct_pref_ev_'+str(ct_ev))




    # update from previous rolling planning timeframe
    cons_ev_init = m.add_constraints(SOC_EV.isel(t=0) == parameters_model["ev_soc_init_abs"], name='cons_ev_init')


    # EV Charging behavior
    cons_ev_charge_home_max = m.add_constraints(P_HOME <= emob_home_xr * parameters_model["ev_p_charge_home"], name='cons_ev_charge_home_max')
    # always allow public charging to prevent infeasibilities
    #cons_ev_charge_public_max = m.add_constraints(P_PUBLIC <= emob_public_xr * parameters_model["ev_p_charge_not_home"], name='cons_ev_charge_public_max') # optional
    cons_ev_charge_public_max = m.add_constraints(P_PUBLIC <= parameters_model["ev_p_charge_not_home"], name='cons_ev_charge_public_max') # optional


    cons_balance = m.add_constraints(P_HOME + P_PUBLIC == P_EV, name='cons_balance')


    # objective terms
    
    # deduce time preference
    timepref_xr = 1*emob_home_xr
    timepref_xr[:] = 0
    
    arrivals_xr = ((emob_home_xr) & (emob_home_xr.shift(t=-1).fillna(emob_home_xr.isel(t=-1))==False))
    arrivals_idx = 1*arrivals_xr*xr.DataArray(range(0,len(arrivals_xr["t"])), dims="t")

    for ct_v in range(0,len(emob_home_xr["v"])): # each vehicle
        arrivals_v = arrivals_idx.isel(v=ct_v).to_numpy()
        arrivals_v_tf = 1*(arrivals_v>0)
        arrivals_v_nz = arrivals_v[arrivals_v>0]
        reduction = arrivals_v_tf
        reduction[reduction>0] = arrivals_v_nz
        reduction = pd.Series(reduction).replace(0, np.nan).ffill().to_numpy()
        reduction[np.isnan(reduction)] = 0
        
        timepref_xr[:,ct_v] = 100 + 10*np.linspace(1,len(emob_home_xr["t"]),len(emob_home_xr["t"])) - 1 * reduction #+ penalize_noon
    
    
    # penalty terms
    cons_timepreference = m.add_constraints(PENALTY_TIMEPREF == (timepref_xr*P_EV).sum("t"), name='cons_penalty_timepref')
    
    cons_no_st_ht_segment_reg = m.add_constraints(PENALTY_ST_HT == parameters_opti["penalty_no_st_ht"] * (emob_STHT_xr * P_EV).sum("t"), name='cons_penalty_no_st_ht_segment_reg')

    cons_no_charge_before_first_arrival = m.add_constraints(PENALTY_NO_CHARGE_BEFORE_ARRIVAL == parameters_opti["penalty_no_charge_before_arrival"] * (emob_true_before_geting_home_xr * P_EV).sum("t"), name='cons_penalty_no_charge_before_arrival')

    obj = C_ALL_planning + 0 + parameters_opti["weight_time_preference"] * PENALTY_TIMEPREF + parameters_opti["weight_only_low_segment"] * PENALTY_ST_HT + parameters_opti["weight_no_charge_before_arrival"] * PENALTY_NO_CHARGE_BEFORE_ARRIVAL


    m.add_objective(obj)
    
    warnings.simplefilter(action='default', category=FutureWarning)
    warnings.simplefilter(action='default', category=UserWarning)      

    
    return m



