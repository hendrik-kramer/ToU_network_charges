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




# save results
string_time = r"2025-10-10_18-24"

result_C_OP_NO_PENALTY_eur = xr.open_dataarray(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\C_OP_NO_PENALTY_"+string_time+".nc")
result_SOC_EV = xr.open_dataarray(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\SOC_EV_"+string_time+".nc")




# ===== plotting ====


if (True):
    
    result_C_OP_NO_PENALTY_eur.mean(dim="r")
    
    species = result_C_OP_NO_PENALTY_eur["r"].to_pandas().to_list()
    penguin_means = {'red': result_C_OP_NO_PENALTY_eur.sel(s='reg').mean(dim="v"),
                    'reg': result_C_OP_NO_PENALTY_eur.sel(s='red').mean(dim="v") }
    
    fig, ax = plt.subplots(layout='constrained')
    
    x = np.arange(len(result_C_OP_NO_PENALTY_eur.mean(dim="v")))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Euro')
    ax.set_title('Cost without penalties')
    ax.set_xticks(x + width)
    ax.set_xticklabels(species, rotation=90)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)



if (True): # EV SOC
    pd_res = result_SOC_EV.isel(v=1, s=1).to_pandas()
    plt.figure()
    plt.plot(pd_res)
    plt.legend(pd_res.columns)
    plt.ylabel("SOC EV in kWh")
    plt.show()


if (True) and parameters_model["settings_setup"] == "prosumage": # P_PV
    plt.figure()
    plt.plot(result_P_PV)
    plt.legend(result_P_PV.columns)
    plt.ylabel("P PV in kW")
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


if (False): # P_BUY
    # quantile plot of EV over all regions
    fig1, ax1 = plt.subplots()
    for quantile in [0.0, 0.1, 0.2, 0.3, 0.4]:
        ax1.fill_between(result_SOC_EV.isel(v=1,s=1).to_pandas().index,
                        result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").quantile(axis=1, q=quantile),
                        result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").quantile(axis=1, q=1-quantile),
                        color='b', alpha= quantile + 0.3, edgecolor=None)
    ax1.plot(result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").median(axis=1), color="k", linewidth=1)
    plt.show()



    buy_scatter = m["P_BUY"].solution.isel(r=1, v=1).to_pandas().rename(columns={"red":"red_buy", "reg":"reg_buy"}).reset_index()
    cost_pd = cost_xr.isel(r=1).to_pandas().reset_index().rename(columns={"red":"red_spot", "reg":"reg_spot"}).reset_index()
    
    merge_result = buy_scatter.merge(cost_pd)
    merge_result["time_of_day"] = merge_result.t.dt.hour + merge_result.t.dt.minute/60
    merge_result = merge_result[merge_result["reg_buy"] > 0 ]
    merge_result["marker_size"] = 25 * merge_result["reg_buy"]
    

    fig, ax = plt.subplots(1, 1)
    merge_result.plot.scatter(x='time_of_day',y='reg_spot', s="marker_size", alpha=0.3, xlabel="Time of the day", ylabel="charge price", ax=ax, c="blue", legend="regular")
    merge_result.plot.scatter(x='time_of_day',y='red_spot', s="marker_size", alpha=0.3, xlabel="Time of the day", ylabel="charge price", ax=ax, c="orange", legend="reduced")
    ax.legend(["regular", "reduced"])

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

