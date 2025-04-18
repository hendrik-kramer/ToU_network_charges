# use linopti environment
# conda activate linopti
# in Spyder: right click on tab main_script.py --> set console working directory

# %matplotlib qt

import sys
print(sys.executable)

import pandas as pd
from linopy import Model
import matplotlib.pyplot as plt 
from datetime import date, timedelta, datetime
import numpy as np
import xarray as xr
import warnings
import os
from pathlib import Path


import functions_tariff_network_charge_study.load_functions as f_load
#import functions_tariff_network_charge_study.model_perfect_foresight_parallel as model_perf_forsight
#import functions_tariff_network_charge_study.model_rolling_foresight_parallel as model_roll_forsight
#import functions_tariff_network_charge_study.model_emob_perfect_foresight_parallel as model_emob_perf_forsight
#import functions_tariff_network_charge_study.model_emob_rolling_foresight_parallel as model_emob_roll_forsight_smart

#import functions_tariff_network_charge_study.model_emob_annual_forsight_parallel as model_emob_annual_forsight_smart
import functions_tariff_network_charge_study.model_emob_annual_forsight_parallel2 as model2


# ===== PARAMETERS ======

# read in data

which_dsos = range(0,100)   # 0 for all, otherwise use range or indices of xlsx file
parameter_year = 2021 
result_folder = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + "\\"


warnings.simplefilter(action='ignore', category=UserWarning)      


# get relevant timesteps to compute KW1-KW52/53
timesteps = f_load.load_timesteps(parameter_year)

# Load spot prices
parameter_folderpath_prices = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\preise" + "\\"
spot_prices_xr = f_load.load_spot_prices(parameter_year, parameter_folderpath_prices, "id_auktion_15_uhr", timesteps) # "da_auktion_12_uhr", "id_auktion_15_uhr" # in ct/kWh
tariff_price = f_load.get_tariff_prices(spot_prices_xr)

# Load network charges (regular and reduced)
parameter_filepath_dsos = r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v3.xlsx"
network_charges_xr = f_load.load_network_charges(parameter_filepath_dsos, which_dsos, timesteps) # dimension: Time x DSO region x scenario (red, reg)

# Load e-Mobility
parameter_folderpath_emob_demand = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_consumption_total_2021_test.csv"
parameter_folderpath_emob_state = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_state_total_2021_test.csv"
emob_demand_xr, emob_state_xr = f_load.load_emob(parameter_folderpath_emob_demand, parameter_folderpath_emob_state, timesteps)

emob_arrival_times, emob_departure_times, dict_idx_lookup = f_load.deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps, 0)   # CAN BE IMPROVED, ONLY FIRST SHOT

# load irradiation // dummy value 
#parameter_folder_irradiance = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation"
parameter_file_hochrechung = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\netztransparenz\Hochrechnung Solarenergie [2025-03-21 13-55-15].csv" # in MW, in UTC
parameter_file_capacities = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\install_capa_controllarea.csv" # in MW

# data not plausible (FLH) of 50hzt! Do not use
irradiance_xr = np.round(f_load.load_irradiance(parameter_file_hochrechung, parameter_file_capacities, timesteps), decimals=3)

# load temperature // use dummy temperature (COSMO-REA6 from 2013) from nodal Flex paper as first guess --> needs to be updated
# parameter_folderpath_temperature = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\temperature\temperature_nodalFlex.csv"
# temperature_xr = f_load.load_temperature(parameter_folderpath_temperature, timesteps)


warnings.simplefilter(action='default', category=UserWarning)      




# Sanity check
if len(emob_demand_xr) != len(spot_prices_xr) or len(emob_demand_xr) != len(network_charges_xr) :
    print("Error: timeseries do not have equal length")


parameters_opti = {
    "settings_obj_fnct": "smart_charging", # "immediate_charging", # "scheduled_charging" "smart_charging"
    "settings_setup": "prosumage", # "only_EV", # "prosumage"
    "rolling_window": "no/year", # "year"
    "quarter" : "Q2",
    "prices": "spot", # "spot", "mean"
    "dso_subset" : range(0,10),
    "emob_subset" : range(0,6),
    "tso_subset" : range(1,2),
    }

parameters_model = {
    "ev_p_charge_home":11, # kW
    "ev_soc_max": 70, # kWh
    "ev_soc_init": 0.9, # %
    "ev_soc_preference": 0.95, # %
    "ev_soc_departure": 0.95, # %
    "ev_p_charge_not_home": 22, # kW
    "ev_eta_in": 0.95,
    "ev_losses": 0.0001,
    "bess_p_max": 5, # kW
    "bess_soc_max": 9, # kWh
    "bess_soc_init": 0.9, # %
    "bess_eta_ch": 0.95, # %
    "bess_eta_dch": 0.95, # %
    "bess_losses": 0.0001, # %
    "pv_p_max": 8 # kW
    }


length_chunk = 10
lst = parameters_opti["dso_subset"]
division = len(lst) / length_chunk
chunks = [lst[round(length_chunk * i):round(length_chunk * (i + 1))] for i in range(int(np.ceil(division)))]

if (parameters_opti["rolling_window"] == "day") and (len(parameters_opti["dso_subset"]) < length_chunk):
    raise ValueError("If rolling on daily basis, no chunks can be used")




#dso_names = network_charges_xr.r.to_numpy()[list(parameters_opti["dso_subset"])]

time_subset = timesteps[timesteps["Quarter"] == parameters_opti["quarter"]].index #range(0,96*30)
dso_subset = parameters_opti["dso_subset"]
emob_subset = parameters_opti["emob_subset"]
tso_subset = parameters_opti["tso_subset"]
   
timesteps = timesteps.iloc[time_subset]
spot_prices_xr = spot_prices_xr.isel(t=time_subset)
network_charges_xr = network_charges_xr.isel(t=time_subset, r=dso_subset)
emob_demand_xr = emob_demand_xr.isel(t=time_subset, v=emob_subset)
emob_state_xr = emob_state_xr.isel(t=time_subset, v=emob_subset)
emob_departure_times = emob_departure_times.iloc[emob_subset]

irradiance_xr = irradiance_xr.isel(t=time_subset, a=tso_subset)

emob_home_xr = (emob_state_xr=="home")
emob_HT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").mean()).drop_vars("s")

timesteps_from_zero = timesteps.reset_index()
_, _, dict_idx_lookup = f_load.deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps, -timesteps.index[0] )   # CAN BE IMPROVED, ONLY FIRST SHOT






for chunk_dso in chunks:

    print("   ")
    print("=== dso chunk: " + str(chunk_dso) + " == " + str(datetime.now()) + " ===")
    print("   ")

    # overwrite parameter for the loop
    parameters_opti["dso_subset"] = chunk_dso
    
    
    
    
    
    # create and run model
    # m = model_emob_annual_forsight_smart.model_emob_annual_smart(timesteps, spot_prices_xr, tariff_prices_xr, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters_model, parameters_opti)
    


    # initialize rolling planning timestep ranges
    if parameters_opti["rolling_window"] == "day":
        unique_days = timesteps["DateTime"].dt.date.unique()
        rolling_timesteps = [range(0, 15*4)] # first day until 3 pm
        for ct_day in unique_days[:-1]: # until second last day
            ct_next_day = ct_day + timedelta(days=1)
            day_min_idx = timesteps_from_zero[(timesteps_from_zero["DateTime"].dt.date == ct_day) & (timesteps_from_zero["DateTime"].dt.hour == 15) & (timesteps_from_zero["DateTime"].dt.minute == 0)].index.item()
            day_max_idx = timesteps_from_zero[(timesteps_from_zero["DateTime"].dt.date == ct_next_day) & (timesteps_from_zero["DateTime"].dt.hour == 23) & (timesteps_from_zero["DateTime"].dt.minute == 45)].index.item()           
            rolling_timesteps.append( range(day_min_idx, day_max_idx+1))
        # no necessity to optimize last half day if overlapping window is known until midnight d+1 anyway
        #rolling_timesteps.append( range(timesteps.iloc[[-9*4]].index.item(), timesteps.iloc[[-1]].index.item()+1)) # last day 3 pm to midnight
        [len(ct) for ct in rolling_timesteps]
    
        first_iteration = True
        soc_ev_last = []
        soc_bess_last = []
        for ct_rolling in rolling_timesteps:
            print("===== time " + str(ct_rolling) + " =====")
            
            timesteps_roll = timesteps.iloc[list(ct_rolling)]
            
            # cut off time after 15 pm of the next day for saving results, but exclude last day
            if ct_rolling[0] < rolling_timesteps[-1][0]:
                timesteps_today = timesteps_roll[timesteps_roll.DateTime <= timesteps_roll.DateTime.iloc[-1] - pd.Timedelta(9, "h")]
            else:
                timesteps_today = timesteps_roll
                                                 
            idx_today_opti = timesteps_today.index - ct_rolling[0] - timesteps.index[0]
    
            # get preferred departure times of the day
            dict_idx_lookup_sub = dict_idx_lookup.copy()
            for key in dict_idx_lookup_sub:
                dict_idx_lookup_sub[key] = dict_idx_lookup[key][dict_idx_lookup[key].isin(timesteps_roll.index)] - timesteps.index[0] - ct_rolling[0]  # reduce by quarter start and first rolling timestep to have indices starting at zero
  
    
            if not(first_iteration):
               parameters_model["ev_soc_init"] = soc_ev_last 
               parameters_model["bess_soc_init"] = soc_bess_last
            else:
               parameters_model["ev_soc_init"] = parameters_model["ev_soc_init"] * parameters_model["ev_soc_max"]
               parameters_model["bess_soc_init"] = parameters_model["bess_soc_init"] * parameters_model["bess_soc_max"]
            
            ct_rolling = timesteps_roll.index - timesteps.index[0]
            spot_prices_xr_roll = spot_prices_xr.isel(t=ct_rolling)
            network_charges_xr_roll = network_charges_xr.isel(t=ct_rolling)
            emob_demand_xr_roll = emob_demand_xr.isel(t=ct_rolling)
            emob_state_xr_roll = emob_state_xr.isel(t=ct_rolling)
            irradiance_xr_roll = irradiance_xr.isel(t=ct_rolling)
            
            # (spot_prices_xr_roll + network_charges_xr_roll).isel(r=1).to_pandas().plot()

            
            
            m = model2.model_emob_annual_smart2(timesteps_roll, spot_prices_xr_roll, tariff_price, network_charges_xr_roll, emob_demand_xr_roll, emob_state_xr_roll, emob_departure_times, dict_idx_lookup_sub, irradiance_xr_roll, parameters_model, parameters_opti)
            m.solve('gurobi', OutputFlag=0, presolve=-1, LogToConsole=0, Method=-1, PreSparsify=-1)

            #if ct_rolling[0] == 1788:
            #    print("Achtung")

            if m.termination_condition == "infeasible":
                #label = m.computeIIS()
                m.print_infeasibilities()

            #if m["P_BUY"].solution.isel(r=1,v=1).sum() > 0:
            #    print("Kauf")

            #if m[0] == "infeasible":
            #    raise  ValueError("Infeasible " + str(ct_rolling))
                    
            if m["SOC_MISSING"].solution.sum().item() > 0:
                print("Slack SOC_MISSING")
                print(m["SOC_MISSING"].solution.sum(dim=["t","r","s"]))
         
            if m["P_EV_NOT_HOME"].solution.sum().item() > 0:
                print("P_EV_NOT_HOME")
                print(m["P_EV_NOT_HOME"].solution.sum(dim=["t","r","s"]))


            if ct_rolling[0] == 0:
                result_C_OP = m["C_OP"].solution
                result_C_OP_NO_PENALTY = m["C_OP_NO_PENALTY"].solution
                result_SOC_EV = m["SOC_EV"].solution
                result_P_BUY =  m["P_BUY"].solution
                result_P_EV_NOT_HOME = m["P_EV_NOT_HOME"].solution
                result_SOC_MISSING = m["SOC_MISSING"].solution
                
                if parameters_opti["settings_setup"] == "prosumage":
                    result_P_PV = m["P_PV"].solution
                    result_SOC_BESS = m["SOC_BESS"].solution

            else:
                result_C_OP = xr.concat([result_C_OP, m["C_OP"].solution], dim="t")
                result_C_OP_NO_PENALTY = xr.concat([result_C_OP_NO_PENALTY, m["C_OP_NO_PENALTY"].solution], dim="t")
                
                # time dependent
                result_SOC_EV = xr.concat([result_SOC_EV, m["SOC_EV"].solution.isel(t=idx_today_opti)], dim="t")
                result_P_BUY = xr.concat([result_P_BUY, m["P_BUY"].solution.isel(t=idx_today_opti)], dim="t")
                result_P_EV_NOT_HOME = xr.concat([result_P_EV_NOT_HOME, m["P_EV_NOT_HOME"].solution.isel(t=idx_today_opti)], dim="t")
                result_SOC_MISSING = xr.concat([result_SOC_MISSING, m["SOC_MISSING"].solution.isel(t=idx_today_opti)], dim="t")
                
                if parameters_opti["settings_setup"] == "prosumage":
                    result_P_PV = xr.concat([result_P_PV, m["P_PV"].solution.isel(t=idx_today_opti)], dim="t")
                    result_SOC_BESS = xr.concat([result_SOC_BESS, m["SOC_BESS"].solution.isel(t=idx_today_opti)], dim="t")
                    
            first_iteration = False
            soc_ev_last = m["SOC_EV"].solution.isel(t=-1)
            
            if parameters_opti["settings_setup"] != "only_EV":
                soc_bess_last = m["SOC_BESS"].solution.isel(t=-1)

    else:
        
        for key in dict_idx_lookup:
            dict_idx_lookup[key] = dict_idx_lookup[key][dict_idx_lookup[key].isin(timesteps.index)]

        m = model2.model_emob_annual_smart2(timesteps_from_zero, spot_prices_xr, tariff_price, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters_model, parameters_opti)
        m_status = m.solve('gurobi', OutputFlag=0, presolve=-1, LogToConsole=0, Method=-1, PreSparsify=-1)
    


    
    
        # ==== DISPLAY if infeasibility variables are positive =====
        
        if m["SOC_MISSING"].solution.sum().item() > 0:
            print("Slack SOC_MISSING")
            print(m["SOC_MISSING"].solution.sum(dim=["t","r","s"]))
    
        if m["P_EV_NOT_HOME"].solution.sum().item() > 0:
            print("P_EV_NOT_HOME")
            print(m["P_EV_NOT_HOME"].solution.sum(dim=["t","r","s"]))
    
    
    
    
        # === store results in large xarray ===
        
        if chunk_dso[0] == 0:
            result_C_OP = m["C_OP"].solution
            result_C_OP_NO_PENALTY = m["C_OP_NO_PENALTY"].solution
            result_SOC_EV = m["SOC_EV"].solution
            result_P_BUY =  m["P_BUY"].solution
            result_P_EV_NOT_HOME = m["P_EV_NOT_HOME"].solution
            result_SOC_MISSING = m["SOC_MISSING"].solution
            
            if parameters_opti["settings_setup"] == "prosumage":
                result_P_PV = m["P_PV"].solution
                result_SOC_BESS = m["SOC_BESS"].solution
                
        else:
            result_C_OP = xr.concat([result_C_OP, m["C_OP"].solution], dim="r")
            result_C_OP_NO_PENALTY = xr.concat([result_C_OP, m["C_OP_NO_PENALTY"].solution], dim="r")
            result_SOC_EV = xr.concat([result_SOC_EV, m["SOC_EV"].solution], dim="r")
            result_P_BUY = xr.concat([result_P_BUY, m["P_BUY"].solution], dim="r")
            result_P_EV_NOT_HOME = xr.concat([result_P_EV_NOT_HOME, m["P_EV_NOT_HOME"].solution], dim="r")
            result_SOC_MISSING = xr.concat([result_SOC_MISSING, m["SOC_MISSING"].solution], dim="r")
    
            
            if parameters_opti["settings_setup"] == "prosumage":
                result_P_PV = xr.concat([result_P_PV, m["P_PV"].solution], dim="r")
                result_SOC_BESS = xr.concat([result_SOC_BESS, m["SOC_BESS"].solution], dim="r")
    
    
# only for debugging relevant
if (False):
    if parameters_opti["prices"] == "spot":
        cost_xr = np.maximum(network_charges_xr + spot_prices_xr,0)
    elif parameters_opti["prices"] == "mean":
        cost_xr = np.maximum(network_charges_xr + tariff_price,0)



#  ==== END OF LOOP =====

# unit conversion
result_C_OP_eur = result_C_OP/100 # ct --> eur
result_C_OP_NO_PENALTY_eur = result_C_OP_NO_PENALTY/100 # ct --> eur

# convert int64 of datetime object to seconds to be able to save netcdf
idx_time = timesteps[timesteps["Quarter"] == parameters_opti["quarter"]].index 
result_P_BUY["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]
result_SOC_EV["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]
result_P_EV_NOT_HOME["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"] 
result_SOC_MISSING["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]  

warnings.simplefilter(action='default', category=FutureWarning)      

# Ergebnis in Euro: (cost_xr * result_P_BUY).mean("v").sum("t")/100


#  ==== SAVE RESULTS =====

# create new folder for results
str_now = datetime.now().strftime("%Y-%d-%d_%H-%M")
folder_path = Path("../daten_results/" + str_now + "_" + parameters_opti["quarter"] + "_" + parameters_opti["settings_obj_fnct"] + "_" + parameters_opti["settings_setup"])
os.makedirs(folder_path, exist_ok=True)

result_C_OP.to_netcdf(folder_path / "C_OP.nc")
result_C_OP_NO_PENALTY_eur.to_netcdf(folder_path / "C_OP_NO_PENALTY.nc")
result_SOC_EV.to_netcdf(folder_path / "SOC_EV.nc")
result_P_BUY.to_netcdf(folder_path / "P_BUY.nc")
result_P_EV_NOT_HOME.to_netcdf(folder_path / "P_EV_NOT_HOME.nc")
result_SOC_MISSING.to_netcdf(folder_path / "SOC_MISSING.nc")

with open(folder_path / "parameters_opti.txt", 'w') as f:
    f.write(pd.DataFrame.from_dict(parameters_opti, orient='index').to_string(header=False, index=True))
    
with open(folder_path / "parameters_model.txt", 'w') as f:
    f.write(pd.DataFrame.from_dict(parameters_model, orient='index').to_string(header=False, index=True))
    
print("Saved data sucessfully to: " + str(folder_path))

