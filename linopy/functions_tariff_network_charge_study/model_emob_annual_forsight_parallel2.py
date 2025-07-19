import os
import sys
import warnings
from linopy import Model
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime as dt


def model_emob_quarter_smart2(timesteps, spot_prices_xr, tariff_price, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters, parameters_opti):

    timesteplength = (timesteps.DateTime.iloc[1] - timesteps.DateTime.iloc[0]).total_seconds()/3600

    print(dt.now())


    # calucalte arrival SOC --> TO BE UPDATED
    soc_preference = parameters["ev_soc_preference"]
    e_ev_init_percent = parameters["ev_soc_init_rel"]
    e_bess_init_percent = parameters["bess_soc_init_rel"]

    
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    warnings.simplefilter(action='ignore', category=UserWarning)      




    emob_home_xr = (emob_state_xr=="home")
    emob_HT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").mean(dim="t")).drop_vars("s")
    
    

    m = Model()
            
    # Definiere Sets
    set_time = pd.Index(timesteps["DateTime"], name="t")
    #print("set_time " , str(list(set_time)))
    set_dso = pd.Index(network_charges_xr["r"].values, name="r")
    set_region = pd.Index(irradiance_xr["a"].values, name="a")
    set_vehicle = pd.Index(emob_demand_xr["v"].values, name="v")
    set_setup = pd.Index(["reg", "red"], name="s")


    # =========== VARIABLES ===============
    
    #OBJ_WITHOUT_PENALTIES = m.add_variables(name='OBJ_WITHOUT_PENALTIES')
    C_OP_ALL = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name='C_OP_ALL')
    C_OP_HOME = m.add_variables(coords=[set_dso, set_vehicle, set_setup], name="C_OP_HOME")

    # EV Battery
    SOC_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='SOC_EV', lower=0) # EV battery state of charge
    P_EV = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV', lower=0) # EV charge power
    P_BUY = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_BUY', lower=0) # EV Mobility
    if parameters_opti["settings_setup"] == "prosumage":
        #BIN_IN = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='BIN_IN', binary=True) # EV Mobility
        P_DCH = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_DCH', lower=0) # EV Mobility
        P_CH = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='P_CH', lower=0) # EV Mobility
        P_PV = m.add_variables(coords=[set_time, set_dso, set_region, set_setup], name='P_PV', lower=0) # EV Mobility
        SOC_BESS = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='SOC_BESS', lower=0) # EV Mobility




    SOC_BELOW_PREF = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='E_BELOW_PREF', lower=0) # EV Mobility
    P_EV_NOT_HOME = m.add_variables(coords=[set_time,set_dso, set_vehicle, set_setup], name='P_EV_NOT_HOME', lower=0) # EV charge power
    #SOC_MISSING = m.add_variables(coords=[set_time, set_dso, set_vehicle, set_setup], name='SOC_MISSING', lower=0) # EV charge power





        
    # =========== CONSTRAINTS ===============

    # EV Battery
    cons_ev_update = m.add_constraints(SOC_EV.isel(t=range(1,len(set_time))) == SOC_EV.isel(t=range(0,len(set_time)-1)) + timesteplength * (P_EV.isel(t=range(0,len(set_time)-1)) + P_EV_NOT_HOME.isel(t=range(0,len(set_time)-1)) - emob_demand_xr.isel(t=range(0,len(set_time)-1))) , name='cons_ev_update')
    #cons_ev_update_last_fix = m.add_constraints(P_EV.isel(t=len(set_time)-1) == 0, name='cons_ev_update_last_fix')
    cons_ev_max_soc = m.add_constraints(SOC_EV <= parameters["ev_soc_max"], name='cons_ev_max_soc')
    cons_ev_charge_ev_max = m.add_constraints(P_EV <= emob_home_xr * parameters["ev_p_charge_home"], name='cons_ev_charge_home')
    cons_ev_charge_not_home = m.add_constraints(P_EV_NOT_HOME <= parameters["ev_p_charge_not_home"], name='cons_ev_charge_not_home')

    cons_ev_init = m.add_constraints(SOC_EV.isel(t=0) == parameters["ev_soc_init_abs"], name='cons_ev_init')


    # dedicated filling level when person usually departs from home
    for ct_ev in set_vehicle.astype(int):
        t_ev = dict_idx_lookup[emob_departure_times[ct_ev]]
        #con_ct_pref_ev = m.add_constraints( SOC_EV.isel(v=ct_ev, t=dict_idx_lookup[emob_departure_times[ct_ev]]) - SOC_MISSING.isel(v=ct_ev, t=t_ev) >= parameters["ev_soc_departure"] * parameters["ev_soc_max"], name='cons_ct_pref_ev_'+str(ct_ev))
        con_ct_pref_ev = m.add_constraints( SOC_EV.isel(v=ct_ev, t=dict_idx_lookup[emob_departure_times[ct_ev]]) >= parameters["ev_soc_departure"] * parameters["ev_soc_max"], name='cons_ct_pref_ev_'+str(ct_ev))





    if parameters_opti["settings_setup"] == "prosumage":
         # IN = increase power in househould = Discharging of BESS
         # OUT = withdrawal of household power = Injection into BESS
         cons_bess_update = m.add_constraints(SOC_BESS.isel(t=range(1,len(set_time))) == SOC_BESS.isel(t=range(0,len(set_time)-1)) + timesteplength * (parameters["bess_eta_ch"]*P_CH.isel(t=range(0,len(set_time)-1)) - 1/parameters["bess_eta_dch"]*P_DCH.isel(t=range(0,len(set_time)-1))) - parameters["bess_losses"], name='cons_bess_update')
         #cons_bess_update_last_fix_in = m.add_constraints(P_DCH.isel(t=len(set_time)-1) == 0, name='cons_bess_update_last_fix_in')
         #cons_bess_update_last_fix_out = m.add_constraints(P_CH.isel(t=len(set_time)-1) == 0, name='cons_bess_update_last_fix_out')
 
         cons_bess_max_P_DCH = m.add_constraints(P_DCH <= parameters["bess_p_max"], name='cons_p_max_in')
         cons_bess_max_P_CH = m.add_constraints(P_CH <= parameters["bess_p_max"], name='cons_p_max_out')
         #cons_bess_max_p_bin_in = m.add_constraints(P_DCH <= BIN_IN * parameters["bess_p_ev"], name='cons_p_max_in')
         #cons_bess_max_p_bin_out = m.add_constraints(P_CH <= (-BIN_IN+1) * parameters["bess_p_ev"], name='cons_p_max_out')
         cons_bess_max_soc = m.add_constraints(SOC_BESS <= parameters["bess_soc_max"], name='cons_bess_max_soc')
         #cons_bess_circle = m.add_constraints(SOC_BESS.isel(t=1) == SOC_BESS.isel(t=len(set_time)-1))
         cons_bess_init = m.add_constraints(SOC_BESS.isel(t=0) == parameters["bess_soc_init_abs"], name='cons_bess_init')

         cons_pv_p_max = m.add_constraints(P_PV <= (irradiance_xr * parameters["pv_p_max"]), name='cons_pv_p_max')

    #if parameters_opti["settings_obj_fnct"] == "scheduled_charging":
    #    cons_no_HT_charge = m.add_constraints(P_BUY <= emob_HT_xr * 999, name='cons_no_HT_charge')      # NOT NICE


    # Home Balance
    if parameters_opti["settings_setup"] == "prosumage":
        cons_balance = m.add_constraints(P_BUY + P_PV - P_CH + P_DCH == P_EV, name='cons_balance')
    else:
        cons_balance = m.add_constraints(P_BUY == P_EV, name='cons_balance')



    # penalize if below preference
    cons_violation_preference  = m.add_constraints(SOC_EV + SOC_BELOW_PREF >= parameters["ev_soc_preference"] * parameters["ev_soc_max"], name='cons_preference_violation')

    
    # PENALTY
    cons_violation_charge_only_home  = m.add_constraints(P_EV_NOT_HOME <= parameters["ev_p_charge_not_home"], name='cons_violation_charge_only_home')
    cons_violation_charge_only_home_last_fix = m.add_constraints(P_EV_NOT_HOME.isel(t=len(set_time)-1) == 0, name="cons_violation_charge_only_home_last_fix")

    if parameters_opti["prices"] == "spot":
        cost_xr = np.maximum(network_charges_xr + spot_prices_xr,0)
    elif parameters_opti["prices"] == "mean":
        cost_xr = np.maximum(network_charges_xr + tariff_price,0)

    cons_cost_all = m.add_constraints(C_OP_ALL == (cost_xr * P_BUY + (parameters["cost_public_charge_pole"] + network_charges_xr.sel(s="reg").mean(["r","t"])).item() * P_EV_NOT_HOME).sum(dims="t") , name="cons_cost_all")
    cons_cost_home = m.add_constraints(C_OP_HOME == (cost_xr * P_BUY).sum(dims="t"), name="cons_cost_home")

    
    #labels = m.compute_infeasibilities()
    #m.print_infeasibilities()  
    
    # zu minimierende Zielfunktion
    if parameters_opti["settings_obj_fnct"] == "immediate_charging":
        
        # create linrange for temporal preference
        timepref_pd = pd.DataFrame(15 * np.linspace( 1, len(set_time), len(set_time) ) + 100, index=network_charges_xr["t"].to_pandas().index, columns=["timepref"])
        timepref_xr = xr.DataArray(timepref_pd["timepref"])
        obj = (timepref_xr * P_BUY).sum() + 999999 * P_EV_NOT_HOME.sum()
    
    elif parameters_opti["settings_obj_fnct"] == "scheduled_charging":
        # create linrange for temporal preference
        timepref_pd = pd.DataFrame(15 * np.linspace( 1, len(set_time), len(set_time) ) + 100, index=network_charges_xr["t"].to_pandas().index, columns=["timepref"])
        timepref_xr = xr.DataArray(timepref_pd["timepref"])
        obj = (timepref_xr * P_BUY).sum() + 9999999*(emob_HT_xr*P_BUY).sum() + 999999 * P_EV_NOT_HOME.sum()
    
    elif parameters_opti["settings_obj_fnct"] == "partfill_charging":
        obj = 10 * SOC_BELOW_PREF.sum() + 999999 * P_EV_NOT_HOME.sum() #+ 999*SOC_MISSING.sum() 
   

    elif parameters_opti["settings_obj_fnct"] == "smart_charging":    
        obj = C_OP_ALL.sum() #+   999999 * P_EV_NOT_HOME.sum() # + 999*SOC_MISSING.sum()
      
    m.add_objective(obj)
    
    warnings.simplefilter(action='default', category=FutureWarning)
    warnings.simplefilter(action='default', category=UserWarning)      

    
    return m



