# use linopti environment
# conda activate linopti
# in Spyder: right click on tab main_script.py --> set console working directory

import sys
print(sys.executable)

import pandas as pd
from linopy import Model
import matplotlib.pyplot as plt 
from datetime import date, timedelta, datetime
import numpy as np
import xarray as xr

import functions_tariff_network_charge_study.load_functions as f_load
#import functions_tariff_network_charge_study.model_perfect_foresight_parallel as model_perf_forsight
#import functions_tariff_network_charge_study.model_rolling_foresight_parallel as model_roll_forsight
#import functions_tariff_network_charge_study.model_emob_perfect_foresight_parallel as model_emob_perf_forsight
#import functions_tariff_network_charge_study.model_emob_rolling_foresight_parallel as model_emob_roll_forsight_smart
import functions_tariff_network_charge_study.model_emob_annual_foresight_parallel as model_emob_annual_forsight_smart

# ===== PARAMETERS ======

# read in network tariffs


which_dsos = range(0,20)   # 0 for all, otherwise use range or indices of xlsx file
parameter_year = 2021


# get relevant timesteps to compute KW1-KW52/53
timesteps = f_load.load_timesteps(parameter_year)

# Load spot prices
parameter_folderpath_prices = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\preise" + "\\"
spot_prices_xr = f_load.load_spot_prices(parameter_year, parameter_folderpath_prices, "id_auktion_15_uhr", timesteps) # "da_auktion_12_uhr", "id_auktion_15_uhr" # in ct/kWh

# Load network charges (regular and reduced)
parameter_filepath_dsos = r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v3.xlsx"
network_charges_xr = f_load.load_network_charges(parameter_filepath_dsos, which_dsos, timesteps) # dimension: Time x DSO region x scenario (red, reg)

# Load e-Mobility
parameter_folderpath_emob_demand = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_consumption_total.csv"
parameter_folderpath_emob_state = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\e_mobility_emoby\ev_state_total.csv"
emob_demand_xr, emob_state_xr = f_load.load_emob(parameter_folderpath_emob_demand, parameter_folderpath_emob_state, timesteps)

emob_arrival_times, emob_departure_times, dict_idx_lookup = f_load.deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps)   # CAN BE IMPROVED, ONLY FIRST SHOT

# load irradiation // dummy value 
#parameter_folder_irradiance = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation"
parameter_file_hochrechung = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\netztransparenz\Hochrechnung Solarenergie [2025-03-21 13-55-15].csv" # in MW, in UTC
parameter_file_capacities = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\irradiation\install_capa_controllarea.csv" # in MW

# data not plausible (FLH) of 50hzt! Do not use
irradiance_xr = np.round(f_load.load_irradiance(parameter_file_hochrechung, parameter_file_capacities, timesteps), decimals=3)

# load temperature // use dummy temperature (COSMO-REA6 from 2013) from nodal Flex paper as first guess --> needs to be updated
# parameter_folderpath_temperature = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\temperature\temperature_nodalFlex.csv"
# temperature_xr = f_load.load_temperature(parameter_folderpath_temperature, timesteps)




# Sanity check
if len(emob_demand_xr) != len(spot_prices_xr) or len(emob_demand_xr) != len(network_charges_xr) :
    print("Error: timeseries do not have equal length")



timesteplength = 0.25 # h

parameters_model = {
    "settings_obj_fnct": "smart_charging", # "immediate_charging", # "scheduled_charging" "smart_charging"
    "settings_setup": "only_EV", # "only_EV", # "prosumage"
    "ev_p_ev":3.7, # kW
    "ev_soc_max": 70, # kWh
    "ev_soc_preference": 1,
    "ev_soc_departure": 1,
    "ev_eta_in": 0.95,
    "ev_losses": 0.0001,
    "stor_p_ev": 5, # kW
    "stor_soc_max": 9, # kWh
    "stor_eta_in": 0.95,
    "stor_eta_out": 0.95,
    "stor_losses": 0.0001,
    "pv_p_max": 8
    }

parameters_opti = {
    "quarter" : "Q1",
    "dso_subset" : range(0,50),
    "emob_subset" : range(0,20),
    "tso_subset" : range(1,2),
    }


length_chunk = 10
lst = parameters_opti["dso_subset"]
division = len(lst) / length_chunk
chunks = [lst[round(length_chunk * i):round(length_chunk * (i + 1))] for i in range(int(np.ceil(division)))]



for chunk_dso in chunks:

    print("   ")
    print("=== chunk: " + str(chunk_dso) + " == " + str(datetime.now()) + " ===")
    print("   ")

    # overwrite parameter
    parameters_opti["dso_subset"] = chunk_dso
    
    # create and run model
    m = model_emob_annual_forsight_smart.model_emob_annual_smart(timesteps, spot_prices_xr, network_charges_xr, emob_demand_xr, emob_state_xr, emob_departure_times, dict_idx_lookup, irradiance_xr, parameters_model, parameters_opti)

    m_status = m.solve('gurobi', OutputFlag=0, presolve=2, LogToConsole=0, Method=1, PreSparsify=2)


    #result_SOC_EV = m["SOC_EV"].solution
    #result_P_BUY = m["P_BUY"].solution
    result_C_OP = m["C_OP"].solution


    
    #if parameters_model["settings_setup"] == "prosumage":
    #    result_P_PV = m["P_PV"].solution.isel(r=1,s=1).to_pandas()
    #    result_SOC_BESS = m["SOC_BESS"].solution.isel(r=1, v=1).to_pandas()
    
    #result_SOC_MISSING = m["SOC_MISSING"].solution.sum().item()


    # === store results ===
    
    if chunk_dso[0] == 0:
        result_C_OP_all = result_C_OP
        #result_SOC_EV_all = result_SOC_EV
        #result_P_BUY_all =  result_P_BUY
    else:
        result_C_OP_all = xr.concat([result_C_OP_all, result_C_OP], dim="r")
        #result_SOC_EV_all = xr.concat([result_SOC_EV_all, result_SOC_EV], "r")
        #result_P_BUY_all = xr.concat([result_P_BUY_all, result_P_BUY], "r")

    del m



if (True): # EV SOC
    plt.figure()
    plt.plot(result_SOC_EV)
    plt.legend(result_SOC_EV.columns)
    plt.ylabel("SOC EV in kWh")

    plt.show()

if (True) and parameters_model["settings_setup"] == "prosumage": # P_PV
    plt.figure()
    plt.plot(result_P_PV)
    plt.legend(result_P_PV.columns)
    plt.ylabel("P PV in kW")
    plt.show()

    plt.show()
    
if (True) and parameters_model["settings_setup"] == "prosumage": # BESS SOC
    plt.figure()
    plt.plot(result_SOC_BESS)
    plt.legend(result_SOC_BESS.columns)
    plt.ylabel("SOC BESS in kWh")
    plt.show()


if (False): # P_BUY
    plt.figure()
    plt.plot(result_P_BUY)
    plt.legend(result_P_BUY.columns)

    plt.show()

#print("result_SOC_MISSING = " + str(result_SOC_MISSING))

#labels = m.compute_infeasibilities()
#m.print_infeasibilities()



#result_cost = (prices_xr * result_p).sum('t').to_pandas()
#einsparung = (result_cost["reg"] - result_cost["red"]) / result_cost["reg"] * 100
#print("Einsparung in Prozent: ", str(einsparung))













# ====== DEDUCE HEAT PUMP DEMAND FROM TEMPERATURE ======


#alpha = 0.0025  # W/(m^2*K)
#surface = 200 #m^2
#limit_temp = 15 # °C

#heat_demand = alpha * surface * np.maximum(limit_temp-temperature_cut, 0) 
#heat_demand_xr = xr.DataArray(heat_demand, dims='t')


#if (False):
#    plt.plot(heat_demand)



# ====== Heat pump parameters =====
#e_max = 20  # kWh
#p_hp = 4 # kW
#cop = 3 # [-]
#timesteplength = 1 # h

# ===== optimization model =====


#m_perf_foresight_det = model_perf_forsight.build_hp_model(prices, dsos, prices_xr, e_max, p_hp, cop, heat_demand_xr, penalty, timesteplength)

#e_init_percent = 0.6
#e_min_end_percent = 0.8




# plots
#if (False):
#    result_P_HP.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="power in kW", xlabel="time")


#if (False):
#    result_E_HStor.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="energy in kWh", xlabel="time")

