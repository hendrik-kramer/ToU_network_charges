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

warnings.simplefilter(action='ignore', category=UserWarning)      



parameters_opti = {
    "year":2024,
    "settings_setup": "only_EV", # "only_EV", # "prosumage"
    "auction": "da_auction_hourly_12_uhr_stairs",  # "da_auction_hourly_12_uhr_linInterpol", "da_auction_hourly_12_uhr_stairs", "da_auction_quarterly_12_uhr", id_auktion_15_uhr"
    "prices": "spot", # "spot", "mean"
    "settings_obj_fnct": "partfill_scheduled_charging", # "immediate_charging", # "scheduled_charging" "partfill_immediate_charging" "partfill_scheduled_charging" "smart_charging"
    "network_charges_sensisitity_study": False,
    "rolling_window": "day", # "no/year", "day"
    "quarter" : "all", # "Q1", "Q2, ...
    "dso_subset" : range(0,30), # excel read in only consideres 100 rows!
    "emob_subset" : range(0,20),
    "tso_subset" : range(4,5),
    }

parameters_model = {
    "ev_p_charge_home":11, # kW
    "ev_soc_max": 70, # kWh
    "ev_soc_init_rel": 0.9, # %
    "ev_soc_preference": 0.8, # %
    "ev_soc_departure": 0.8, # %
    "ev_p_charge_not_home": 22, # kW
    "ev_eta_in": 0.95,
    "ev_losses": 0.0001,
    "bess_p_max": 5, # kW
    "bess_soc_max": 9, # kWh
    "bess_soc_init_rel": 0.9, # %
    "bess_eta_ch": 0.95, # %
    "bess_eta_dch": 0.95, # %
    "bess_losses": 0.01, # %
    "pv_p_max": 8, # kW
    "cost_public_charge_pole": 55 # ct/kW   # https://de.statista.com/statistik/daten/studie/882563/umfrage/strompreise-an-e-auto-ladesaeulen-nach-betreiber-in-deutschland/
    }


network_drive = r"\\wiwinf-file01.wiwinf.uni-due.de\home\hendrik.kramer" # r"Z:" 


# get relevant timesteps to compute KW1-KW52/53
timesteps = f_load.load_timesteps(parameters_opti["year"])

# Load spot prices
parameter_folderpath_prices = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\preise" + "\\"
spot_prices_xr = f_load.load_spot_prices(parameters_opti["year"], parameter_folderpath_prices, parameters_opti["auction"], timesteps)  # in ct/kWh
tariff_static_price = f_load.get_annual_static_tariff_prices(spot_prices_xr)

# Load network charges (regular and reduced)
parameter_filepath_dsos = network_drive + r"\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v4.xlsx"
network_charges_xr, xr_dso_quarters_sum, xr_ht_length, xr_nt_length, xr_ht_charge, xr_st_charge, xr_nt_charge = f_load.load_network_charges(parameter_filepath_dsos, timesteps) # dimension: Time x DSO region x scenario (red, reg)

# Load e-Mobility
parameter_folderpath_emob_demand = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_consumption_2025_07_08.csv"
parameter_folderpath_emob_state = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_state_2025_07_08.csv"
emob_demand_xr, emob_state_xr = f_load.load_emob(parameter_folderpath_emob_demand, parameter_folderpath_emob_state, timesteps)

emob_arrival_times, emob_departure_times, dict_idx_lookup = f_load.deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps, 0)   # CAN BE IMPROVED, ONLY FIRST SHOT
# (pd.DataFrame(emob_departure_times, columns=["dept_time"]).groupby("dept_time").size()/len(emob_departure_times)).plot()


# load irradiation // dummy value 
#parameter_folder_irradiance = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\irradiation"
parameter_file_hochrechung = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\netztransparenz\Hochrechnung Solarenergie [2025-03-21 13-55-15].csv" # in MW, in UTC
parameter_file_capacities = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\install_capa_controllarea_mid_year.csv" # in MW
parameter_file_fulloadhours = network_drive + r"\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\mifri_pv_fullloadhours.csv" # in MW, in UTC


# data not plausible (FLH) of 50hzt! Do not use
irradiance_xr = f_load.load_irradiance(parameter_file_hochrechung, parameter_file_capacities, parameter_file_fulloadhours, timesteps)

# load temperature // use dummy temperature (COSMO-REA6 from 2013) from nodal Flex paper as first guess --> needs to be updated
# parameter_foilderpath_temperature = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\temperature\temperature_nodalFlex.csv"
# temperature_xr = f_load.load_temperature(parameter_folderpath_temperature, timesteps)

warnings.simplefilter(action='default', category=UserWarning)      


# Sanity check
if (len(emob_demand_xr) != len(spot_prices_xr)) or (len(emob_demand_xr) != len(network_charges_xr)) :
    print("Warning: timeseries do not have equal length")






length_dso_chunk = 5
lst = parameters_opti["dso_subset"]
division = len(lst) / length_dso_chunk
list_of_dso_chunks = [lst[round(length_dso_chunk * i):round(length_dso_chunk * (i + 1))] for i in range(int(np.ceil(division)))]


# limit data to relevant values according to parameters mentioned above
if parameters_opti["quarter"] != "all":
    time_subset = timesteps[timesteps["Quarter"] == parameters_opti["quarter"]].index 
else:
    time_subset = timesteps[timesteps.DateTime.dt.year==parameters_opti["year"]].index   # irrelevant, when timesteps is loaded, this is already accounted for

dso_subset = parameters_opti["dso_subset"]
emob_subset = parameters_opti["emob_subset"]
tso_subset = parameters_opti["tso_subset"]
timesteps = timesteps.loc[time_subset]
spot_prices_xr = spot_prices_xr.isel(t=time_subset).fillna(10)
network_charges_xr = network_charges_xr.isel(t=time_subset, r=dso_subset)
emob_demand_xr = emob_demand_xr.isel(t=time_subset, v=emob_subset)
emob_state_xr = emob_state_xr.isel(t=time_subset, v=emob_subset)
emob_departure_times = emob_departure_times.iloc[emob_subset]
irradiance_xr = irradiance_xr.isel(t=time_subset, a=tso_subset)


timesteps_from_zero = timesteps.reset_index()
_, _, dict_idx_lookup = f_load.deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps, 0 )   # CAN BE IMPROVED, ONLY FIRST SHOT






for chunk_dso in list_of_dso_chunks:

   
    # overwrite parameter for the loop
    parameters_opti["dso_subset"] = chunk_dso
    

    # initialize rolling planning timestep ranges
    if parameters_opti["rolling_window"] == "day":
        unique_days = timesteps["DateTime"].dt.date.unique()
        
        rolling_timesteps = [range(0, 24*4)] # minor error as after 1st day: soc 23:45 == 0:00 2nd day, otherwise no issue due to overlapping period
        for ct_day in unique_days[:-1]: # until second last day (since "shorter" last day is irrelevant due to no new information)
            ct_next_day = ct_day + timedelta(days=1)
            day_min_idx = timesteps_from_zero[(timesteps_from_zero["DateTime"].dt.date == ct_day) & (timesteps_from_zero["DateTime"].dt.hour == 13) & (timesteps_from_zero["DateTime"].dt.minute == 0)].index.item()
            day_max_idx = timesteps_from_zero[(timesteps_from_zero["DateTime"].dt.date == ct_next_day) & (timesteps_from_zero["DateTime"].dt.hour == 23) & (timesteps_from_zero["DateTime"].dt.minute == 45)].index.item()           
            rolling_timesteps.append( range(day_min_idx, day_max_idx+1))
        
        # no necessity to optimize last half day if overlapping window is known until midnight d+1 anyway
        #rolling_timesteps.append( range(timesteps.iloc[[-9*4]].index.item(), timesteps.iloc[[-1]].index.item()+1)) # last day 3 pm to midnight
        [len(ct) for ct in rolling_timesteps]
    
        first_iteration = True
        soc_ev_last = []
        soc_bess_last = []
        for ct_rolling in rolling_timesteps:
            print("===== dso_chunk: " + str(chunk_dso) + ", time: " + str(ct_rolling) + " =====")
            
            timesteps_roll = timesteps.iloc[list(ct_rolling)]
            timesteps_roll.is_copy = False # to avoid many warnings that modifying "timesteps_roll" does not affect parent "timesteps"
            
            # cut off time after 13 pm of the next day for saving results, but exclude last day
            if ct_rolling[0] < rolling_timesteps[-1][0]: # "this start time is smaller than last day's start time"
                timesteps_today = timesteps_roll[timesteps_roll.DateTime <= timesteps_roll.DateTime.iloc[-1] - pd.Timedelta(11, "h")]
            else:
                timesteps_today = timesteps_roll
                                                 
            idx_today_opti = timesteps_today.index - ct_rolling[0] - timesteps.index[0]
    
            # get preferred departure times of the day
            dict_idx_lookup_sub = dict_idx_lookup.copy()
            for key in dict_idx_lookup_sub:
                dict_idx_lookup_sub[key] = dict_idx_lookup[key][dict_idx_lookup[key].isin(timesteps_roll.index)] - timesteps.index[0] - ct_rolling[0]  # reduce by quarter start and first rolling timestep to have indices starting at zero
  
    
            if not(first_iteration):
               parameters_model["ev_soc_init_abs"] = np.minimum(soc_ev_last , parameters_model["ev_soc_max"]) # numerical errors occured: soc:ev_last = ev_soc_max + 0.0000000001
               parameters_model["bess_soc_init_abs"] = np.minimum(soc_bess_last, parameters_model["bess_soc_max"]) # numerical errors occured: soc_bess_last = bess_soc_max + 0.0000000001
            else:
               parameters_model["ev_soc_init_abs"] = parameters_model["ev_soc_init_rel"] * parameters_model["ev_soc_max"]
               parameters_model["bess_soc_init_abs"] = parameters_model["bess_soc_init_rel"] * parameters_model["bess_soc_max"]
            
            ct_rolling = timesteps_roll.index - timesteps.index[0]
            spot_prices_xr_roll = spot_prices_xr.isel(t=ct_rolling)
            network_charges_xr_roll = network_charges_xr.isel(t=ct_rolling).isel(r=chunk_dso)
            emob_demand_xr_roll = emob_demand_xr.isel(t=ct_rolling)
            emob_state_xr_roll = emob_state_xr.isel(t=ct_rolling)
            irradiance_xr_roll = irradiance_xr.isel(t=ct_rolling)
            
            # (spot_prices_xr_roll + network_charges_xr_roll).isel(r=1).to_pandas().plot()

            # storage rolling takes place each day: pass 13:00 value as init soc for next day 
            pd.set_option("mode.chained_assignment", None)  
            timesteps_roll.loc[:,"counter_id"] = list(range(0,len(timesteps_roll)))  
            pd.set_option("mode.chained_assignment", "warn")    

            idx_to_roll = timesteps_roll[(timesteps_roll.DateTime.dt.hour==13) & (timesteps_roll.DateTime.dt.minute==00)].iloc[-1]
            #warnings.simplefilter(action='ignore', category="SettingWithCopyWarning")      
            timesteps_roll.loc[:,"save_day_data"] = (timesteps_roll.counter_id < idx_to_roll.counter_id)
            #warnings.simplefilter(action='default', category="SettingWithCopyWarning")      

            
            # ===== OPTIMIZE ====
            m = model2.model_emob_quarter_smart2(timesteps_roll, spot_prices_xr_roll, tariff_static_price, network_charges_xr_roll, emob_demand_xr_roll, emob_state_xr_roll, emob_departure_times, dict_idx_lookup_sub, irradiance_xr_roll, parameters_model, parameters_opti)
            m.solve('gurobi', OutputFlag=0, presolve=-1, LogToConsole=0, Method=-1, PreSparsify=-1)


            if m.termination_condition == "infeasible":
                #label = m.computeIIS()
                m.print_infeasibilities()
    
            #if m["SOC_MISSING"].solution.sum().item() > 0:
            #    print("Slack SOC_MISSING")
            #    print(m["SOC_MISSING"].solution.sum(dim=["t","r","s"]))
         
            if m["P_EV_NOT_HOME"].solution.sum().item() > 0:
                print("P_EV_NOT_HOME")
                print(m["P_EV_NOT_HOME"].solution.sum(["t"]).mean(dim=["r","s"]))




    
            soc_ev_last = m["SOC_EV"].solution.isel(t=idx_to_roll.counter_id)
            if parameters_opti["settings_setup"] != "only_EV":
                soc_bess_last = m["SOC_BESS"].solution.isel(t=idx_to_roll.counter_id)
                    
                    

            # save daily results to common data structures of whole time horizon
            if first_iteration:
                result_C_OP_ALL_roll = m["C_OP_ALL"].solution
                result_C_OP_HOME_roll = m["C_OP_HOME"].solution
                result_SOC_EV_roll = m["SOC_EV"].solution.isel(t=range(0,idx_to_roll.counter_id)) # slight error, as only values including 12:45 are copied ...
                result_P_BUY_roll =  m["P_BUY"].solution.isel(t=range(0,idx_to_roll.counter_id))  # ... are also used as initial soc for next optimization, see above.
                result_P_EV_NOT_HOME_roll = m["P_EV_NOT_HOME"].solution.isel(t=range(0,idx_to_roll.counter_id))
                #result_SOC_MISSING_roll = m["SOC_MISSING"].solution.isel(t=range(0,idx_to_roll.counter_id))
                
                if parameters_opti["settings_setup"] == "prosumage":
                    result_P_PV_roll = m["P_PV"].solution.isel(t=range(0,idx_to_roll.counter_id))
                    result_SOC_BESS_roll = m["SOC_BESS"].solution.isel(t=range(0,idx_to_roll.counter_id))

            else:
                result_C_OP_ALL_roll = result_C_OP_ALL_roll + m["C_OP_ALL"].solution
                result_C_OP_HOME_roll = result_C_OP_HOME_roll + m["C_OP_HOME"].solution
                result_SOC_EV_roll = xr.concat([result_SOC_EV_roll, m["SOC_EV"].solution.isel(t=idx_today_opti)], dim="t")
                result_P_BUY_roll = xr.concat([result_P_BUY_roll, m["P_BUY"].solution.isel(t=idx_today_opti)], dim="t")
                result_P_EV_NOT_HOME_roll = xr.concat([result_P_EV_NOT_HOME_roll, m["P_EV_NOT_HOME"].solution.isel(t=idx_today_opti)], dim="t")
                #result_SOC_MISSING_roll = xr.concat([result_SOC_MISSING_roll, m["SOC_MISSING"].solution.isel(t=idx_today_opti)], dim="t")
                
                if parameters_opti["settings_setup"] == "prosumage":
                    result_P_PV_roll = xr.concat([result_P_PV_roll, m["P_PV"].solution.isel(t=idx_today_opti)], dim="t")
                    result_SOC_BESS_roll = xr.concat([result_SOC_BESS_roll, m["SOC_BESS"].solution.isel(t=idx_today_opti)], dim="t")
                    
            # roll over soc values of subsequent first timestep
            
            
            first_iteration = False

            
                
        # add dso chunks together
        if chunk_dso[0] == 0:
            result_C_OP_ALL = result_C_OP_ALL_roll
            result_C_OP_HOME = result_C_OP_HOME_roll
            result_SOC_EV = result_SOC_EV_roll
            result_P_BUY =  result_P_BUY_roll
            result_P_EV_NOT_HOME = result_P_EV_NOT_HOME_roll
            #result_SOC_MISSING = result_SOC_MISSING_roll
            
            if parameters_opti["settings_setup"] == "prosumage":
                result_P_PV = result_P_PV_roll
                result_SOC_BESS = result_SOC_BESS_roll
                
        else:
            result_C_OP_ALL = xr.concat([result_C_OP_ALL, result_C_OP_ALL_roll], dim="r")
            result_C_OP_HOME = xr.concat([result_C_OP_HOME, result_C_OP_HOME_roll], dim="r")
            result_SOC_EV = xr.concat([result_SOC_EV, result_SOC_EV_roll], dim="r")
            result_P_BUY = xr.concat([result_P_BUY, result_P_BUY_roll], dim="r")
            result_P_EV_NOT_HOME = xr.concat([result_P_BUY, result_P_EV_NOT_HOME_roll], dim="r")
            #result_SOC_MISSING = xr.concat([result_SOC_MISSING, result_SOC_MISSING_roll], dim="r")

            
            if parameters_opti["settings_setup"] == "prosumage":
                result_P_PV = xr.concat([result_P_PV, result_P_PV_roll], dim="r")
                result_SOC_BESS = xr.concat([result_SOC_BESS, result_SOC_BESS_roll], dim="r")
        
    
    # ====perfect foresight optimization  ====        
    else:
        
        for key in dict_idx_lookup:
            dict_idx_lookup[key] = dict_idx_lookup[key][dict_idx_lookup[key].isin(timesteps.index)]

        m = model2.model_emob_quarter_smart2(timesteps_from_zero, spot_prices_xr, tariff_price, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters_model, parameters_opti)
        m_status = m.solve('gurobi', OutputFlag=0, presolve=-1, LogToConsole=0, Method=-1, PreSparsify=-1)
    

        # display infeasibility variables if applicable
        #if m["SOC_MISSING"].solution.sum().item() > 0:
        #    print("Slack SOC_MISSING")
        #    print(m["SOC_MISSING"].solution.sum(dim=["t","r","s"]))
    
        if m["P_EV_NOT_HOME"].solution.sum().item() > 0:
            print("P_EV_NOT_HOME")
            print(m["P_EV_NOT_HOME"].solution.sum(dim=["t","r","s"]))
    

        #  store total time horizon results in large dataset with all dsos 
        if chunk_dso[0] == 0:
            result_C_OP_ALL = m["C_OP_ALL"].solution
            result_C_OP_HOME = m["C_OP_HOME"].solution
            result_SOC_EV = m["SOC_EV"].solution
            result_P_BUY =  m["P_BUY"].solution
            result_P_EV_NOT_HOME = m["P_EV_NOT_HOME"].solution
            #result_SOC_MISSING = m["SOC_MISSING"].solution
            
            if parameters_opti["settings_setup"] == "prosumage":
                result_P_PV = m["P_PV"].solution
                result_SOC_BESS = m["SOC_BESS"].solution
                
        else:
            result_C_OP_ALL = xr.concat([result_C_OP_ALL, m["C_OP_ALL"].solution], dim="r")
            result_C_OP_HOME = xr.concat([result_C_OP_HOME, m["C_OP_HOME"].solution], dim="r")
            result_SOC_EV = xr.concat([result_SOC_EV, m["SOC_EV"].solution], dim="r")
            result_P_BUY = xr.concat([result_P_BUY, m["P_BUY"].solution], dim="r")
            result_P_EV_NOT_HOME = xr.concat([result_P_EV_NOT_HOME, m["P_EV_NOT_HOME"].solution], dim="r")
            #result_SOC_MISSING = xr.concat([result_SOC_MISSING, m["SOC_MISSING"].solution], dim="r")
    
            
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
result_C_OP_ALL_eur = result_C_OP_ALL/100 # ct --> eur
result_C_OP_HOME_eur = result_C_OP_HOME/100 # ct --> eur

# convert int64 of datetime object to seconds to be able to save netcdf

if parameters_opti["quarter"] != "all":
    idx_time = timesteps[timesteps["Quarter"] == parameters_opti["quarter"]].index 
else:
    idx_time = timesteps[timesteps.DateTime.dt.year==parameters_opti["year"]].index   # can be improved, load emobpy data for iso calendar, not only exact 2024

result_P_BUY_1970 = result_P_BUY
result_P_BUY_1970["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]
result_SOC_EV_1970 = result_SOC_EV
result_SOC_EV_1970["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]
result_P_EV_NOT_HOME_1970 = result_P_EV_NOT_HOME
result_P_EV_NOT_HOME_1970["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"] 
#result_SOC_MISSING_1970 = result_SOC_MISSING
#result_SOC_MISSING_1970["t"] = timesteps.loc[idx_time,"seconds_since_1970_in_utc"]  

warnings.simplefilter(action='default', category=FutureWarning)      

# Ergebnis in Euro: (cost_xr * result_P_BUY).mean("v").sum("t")/100


#  ==== SAVE RESULTS =====

# create new folder for results
str_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
folder_path = Path("../daten_results/" + str_now + "_" + parameters_opti["quarter"] + "_" + parameters_opti["prices"] + "_" + parameters_opti["settings_obj_fnct"] + "_" + parameters_opti["settings_setup"])
os.makedirs(folder_path, exist_ok=True)

result_C_OP_ALL_eur.to_netcdf(folder_path / "C_OP_ALL.nc")
result_C_OP_HOME_eur.to_netcdf(folder_path / "C_OP_HOME.nc")
result_SOC_EV_1970.to_netcdf(folder_path / "SOC_EV.nc")
result_P_BUY_1970.to_netcdf(folder_path / "P_BUY.nc")
result_P_EV_NOT_HOME_1970.to_netcdf(folder_path / "P_EV_NOT_HOME.nc")
#result_SOC_MISSING_1970.to_netcdf(folder_path / "SOC_MISSING.nc")

with open(folder_path / "parameters_opti.txt", 'w') as f:
    f.write(pd.DataFrame.from_dict(parameters_opti, orient='index').to_string(header=False, index=True))
    
parameters_model_wo_2D = parameters_model
del parameters_model_wo_2D['ev_soc_init_abs']
with open(folder_path / "parameters_model.txt", 'w') as f:
    f.write(pd.DataFrame.from_dict(parameters_model_wo_2D, orient='index').to_string(header=False, index=True))
    
print("Saved data sucessfully to: " + str(folder_path))

