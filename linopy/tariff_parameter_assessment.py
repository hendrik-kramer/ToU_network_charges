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
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn import linear_model



# Example usage:
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])


folder_path = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" +  r"\\"

#file_new_contract = r"2025-07-10_11-32_all_spot_scheduled_charging_only_EV_r50_v10" +r"\\"
file_new_static = r"2025-07-12_07-32_all_mean_smart_charging_only_EV_r100_v50"  + r"\\"
file_baseline_static = r"2025-07-13_08-51_all_mean_immediate_charging_only_EV_r100_v50" + r"\\"
file_new_dynamic = r"2025-07-13_01-49_all_spot_smart_charging_only_EV_r100_v50"  + r"\\"
file_baseline_dynamic = r"2025-07-14_10-23_all_spot_immediate_charging_only_EV_r100_v50" + r"\\"

result_P_BUY_old_static = xr.open_dataarray(folder_path + file_baseline_static + "P_BUY.nc")
result_P_BUY_new_static = xr.open_dataarray(folder_path + file_new_static + "P_BUY.nc")
result_P_BUY_old_dynamic = xr.open_dataarray(folder_path + file_baseline_dynamic + "P_BUY.nc")
result_P_BUY_new_dynamic = xr.open_dataarray(folder_path + file_new_dynamic + "P_BUY.nc")


# reconvert seconds to datetime
epoch_time = datetime(1970, 1, 1)
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(result_P_BUY_old["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
result_P_BUY_old["t"] = dti
result_P_BUY_new["t"] = dti

result_P_BUY_old_static = result_P_BUY_old_static.sel(s="reg")
result_P_BUY_new_static = result_P_BUY_new_static.sel(s="reg")
P_delta_neg_static = - np.minimum((result_P_BUY_new_static - result_P_BUY_old_static),0) # decrease of power consumption new < old, but negate to get positive values

result_P_BUY_old_dynamic = result_P_BUY_old_dynamic.sel(s="reg")
result_P_BUY_new_dynamic = result_P_BUY_new_dynamic.sel(s="reg")
P_delta_neg_dynamic = - np.minimum((result_P_BUY_new_dynamic - result_P_BUY_old_static),0) # decrease of power consumption new < old, but negate to get positive values




amount_days = P_delta_neg_static["t"].size / 96

emob_HT_xr = (network_charges_xr.sel(s="red")>network_charges_xr.sel(s="red").mean(dim="t")).drop_vars("s")
emob_STNT_xr = np.logical_not(emob_HT_xr)
xr_dso_quarters_sum = xr_dso_quarters_sum




if (False):
    price_ht_minus_st = (xr_ht_charge - xr_st_charge).to_numpy()
    hours_ht = xr_ht_length.to_numpy()
    hours_nt = xr_nt_length.to_numpy()
    
    
    # ==== SET VARIABLES HERE =========
    
    x_variable = price_ht_minus_st
    y_variable = hours_ht
    
    # Only input correlation:
    fig, ax_test1 = plt.subplots()
    
    ax_test1.scatter( x_variable,y_variable )
    ax_test1.set_xlabel("price difference in €: High - Standard")
    ax_test1.set_ylabel("High segment length in hours" )
        
    reg3 = linear_model.LinearRegression()
    reg3.fit(x_variable.reshape(-1, 1), y_variable.reshape(-1, 1))
    reg3.coef_
    reg3.intercept_
    r2_3 = reg3.score(x_variable.reshape(-1, 1), y_variable.reshape(-1, 1))
    
    x_line = np.linspace(x_variable.min(), x_variable.max(), 100).reshape(-1, 1)
    y_line = reg3.predict(x_line)
    ax_test1.plot(x_line, y_line, color='red', label='Regression line')
    



# sum up all power reductions that occur during HT, then divide by amount of quaters
# factor 1/4 to get from quarterly kw to hourly kwh
average_kWh_neg_kwh_in_ht_per_day_static = ((P_delta_neg_static * emob_HT_xr).sum(dim="t") * (1/4) / (xr_dso_quarters_sum/4 *amount_days)).to_pandas()  
average_kWh_neg_kwh_in_ht_per_day_dynamic = ((P_delta_neg_dynamic * emob_HT_xr).sum(dim="t") * (1/4) / (xr_dso_quarters_sum/4 *amount_days)).to_pandas()  


ht_kwh_shifted_vector_mean_static = np.array(average_kWh_neg_kwh_in_ht_per_day_static).reshape(-1,)
ht_kwh_shifted_vector_mean_dynamic.array(average_kWh_neg_kwh_in_ht_per_day_dynamic).reshape(-1,)


ht_length_vector =  np.repeat(xr_ht_length.to_numpy(), len(average_kWh_neg_kwh_in_ht_per_day.columns))
ht_price_factor_vector =  np.repeat((xr_ht_charge / xr_st_charge).to_numpy(), len(average_kWh_neg_kwh_in_ht_per_day.columns))


use_mean = False
if (use_mean):
    ht_kwh_shifted_vector_mean_static = average_kWh_neg_kwh_in_ht_per_day_static.mean(axis=1).to_numpy()
    ht_kwh_shifted_vector_mean_dynamic = average_kWh_neg_kwh_in_ht_per_day_dynamic.mean(axis=1).to_numpy()

    ht_length_vector = xr_ht_length.to_numpy()
    ht_price_factor_vector = xr_ht_div_st_fraction.to_numpy()



fig_tariff_regression, axs_tariff_regression = plt.subplots(ncols=4, figsize=(15, 5))  # 2 Subplots nebeneinander

# Subplot 1: Scatterplot auf erster Achse
reg = linear_model.LinearRegression()
reg.fit(ht_length_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_static.reshape(-1, 1))
reg.coef_
reg.intercept_
r2 = reg.score(ht_length_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_static.reshape(-1, 1))
x_line = np.linspace(ht_length_vector.min(), ht_length_vector.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
axs_tariff_regression[0].title.set_text("Static Electricity Price")
axs_tariff_regression[0].grid(which='major', axis='both', linestyle='--', color="lightgray", zorder=0)
axs_tariff_regression[0].plot(x_line, y_line, color='#8b3003', label='Regression line', linestyle="--", linewidth=2)
axs_tariff_regression[0].scatter(ht_length_vector, ht_kwh_shifted_vector_mean_static, alpha=0.1, color="#00386c", zorder=2, s=6)
axs_tariff_regression[0].set_xlabel('High segment duration in hours')
axs_tariff_regression[0].set_ylabel('mean daily kWh shifted away from HT')


# Subplot 2: Scatterplot auf zweiter Achse
reg2 = linear_model.LinearRegression()
reg2.fit(ht_price_factor_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_static.reshape(-1, 1))
reg2.coef_
reg2.intercept_
r2 = reg.score(ht_price_factor_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_static.reshape(-1, 1))
x_line = np.linspace(ht_price_factor_vector.min(), ht_price_factor_vector.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
axs_tariff_regression[1].title.set_text("Static Electricity Price")
axs_tariff_regression[1].grid(which='major', axis='both', linestyle='--', color="lightgray", zorder=0)
axs_tariff_regression[1].plot(x_line, y_line, color='#8b3003', label='Regression line', linestyle="--", linewidth=2)
axs_tariff_regression[1].scatter(ht_price_factor_vector, ht_kwh_shifted_vector_mean_static, alpha=0.1, color="#00386c", zorder=2, s=6)
axs_tariff_regression[1].set_xlabel('HT/ST Price')
axs_tariff_regression[1].set_ylabel('mean daily kWh shifted away from HT')


# Subplot 1: Scatterplot auf erster Achse
reg = linear_model.LinearRegression()
reg.fit(ht_length_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_dynamic.reshape(-1, 1))
reg.coef_
reg.intercept_
r2 = reg.score(ht_length_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_dynamic.reshape(-1, 1))
x_line = np.linspace(ht_length_vector.min(), ht_length_vector.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
axs_tariff_regression[2].title.set_text("Dynamic Electricity Price")
axs_tariff_regression[2].grid(which='major', axis='both', linestyle='--', color="lightgray", zorder=0)
axs_tariff_regression[2].plot(x_line, y_line, color='#8b3003', label='Regression line', linestyle="--", linewidth=2)
axs_tariff_regression[2].scatter(ht_length_vector, ht_kwh_shifted_vector_mean_dynamic, alpha=0.1, color="#00386c", zorder=2, s=6)
axs_tariff_regression[2].set_xlabel('High segment duration in hours')
axs_tariff_regression[2].set_ylabel('mean daily kWh shifted away from HT')


# Subplot 2: Scatterplot auf zweiter Achse
reg2 = linear_model.LinearRegression()
reg2.fit(ht_price_factor_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_dynamic.reshape(-1, 1))
reg2.coef_
reg2.intercept_
r2 = reg.score(ht_price_factor_vector.reshape(-1, 1), ht_kwh_shifted_vector_mean_dynamic.reshape(-1, 1))
x_line = np.linspace(ht_price_factor_vector.min(), ht_price_factor_vector.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
axs_tariff_regression[3].title.set_text("Dynamic Electricity Price")
axs_tariff_regression[3].grid(which='major', axis='both', linestyle='--', color="lightgray", zorder=0)
axs_tariff_regression[3].plot(x_line, y_line, color='#8b3003', label='Regression line', linestyle="--", linewidth=2)
axs_tariff_regression[3].scatter(ht_price_factor_vector, ht_kwh_shifted_vector_mean_dynamic, alpha=0.1, color="#00386c", zorder=2, s=6)
axs_tariff_regression[3].set_xlabel('HT/ST Price')
axs_tariff_regression[3].set_ylabel('mean daily kWh shifted away from HT')





plt.tight_layout()
plt.show()


if (False):
    plt.plot(ht_kwh_shifted_vector)
    plt.plot(ht_length_vector)
    plt.plot(ht_price_factor_vector)
