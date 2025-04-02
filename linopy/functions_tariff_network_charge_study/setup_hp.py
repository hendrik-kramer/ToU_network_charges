# use linopti environment
# conda activate linopti

import sys
print(sys.executable)

import pandas as pd
import numpy as np

from linopy import Model
import xarray as xr

import matplotlib.pyplot as plt 


# ===== PARAMETERS ======

netzbetreiber = ["N-ERGIE Netz", "Mainfranken Netze", "WEMAG Netz","Allgäu Netz","badenovaNETZE","Netze ODR","Harz Energie Netz","LSW Netz","Westfalen Weser Netz","Energie Waldeck-Frankenberg","SachsenNetze HS.HD","Avacon Netz","Celle-Uelzen Netz","Schleswig-Holstein Netz","e-netz Südhessen","Pfalzwerke Netz","energis-Netzgesellschaft","Energienetze Mittelrhein"]


parameter_year = 2021
param_price = "red"


penalty = 9999


# ====== ELECTRICITY PRICES - HISTORIC WHOLESALE PRICES =====

# read in time data 
raw_price_data = pd.read_csv(r"Z:\10_Paper\13_Alleinautorenpaper\DayAheadPrices_12_1_D_2019_2024_hourly_quarterly.csv")

# convert utc time to local time
raw_price_data['Time_utc'] = pd.to_datetime(raw_price_data['DateTime'], format='%Y-%m-%d %H:%M:%S', utc=True)
raw_price_data['Time_DE'] = raw_price_data['Time_utc'].dt.tz_convert('Europe/Berlin')
raw_price_data['iso_year'] = raw_price_data['Time_DE'].dt.isocalendar().year
raw_price_data['iso_week'] = raw_price_data['Time_DE'].dt.isocalendar().week
raw_price_data["Value"] = raw_price_data["Value"]/10 # €/MWh --> ct/kWh


# select relevant year and treat daylight saving time jumps
# Script tries to select 52 or 53 whole weeks accoring to iso week definition, unless half week input data in neighboring years is missing

price_data = raw_price_data[(raw_price_data['iso_year']==parameter_year) & (raw_price_data['ResolutionCode']=="PT15M")]
price_data["DateTime"] = pd.to_datetime(price_data["Time_DE"])
price_series = pd.DataFrame(pd.date_range(pd.to_datetime(str(parameter_year) + "-01-01"), pd.to_datetime(str(parameter_year) + "-12-31"), freq="1h", tz="Europe/Berlin"), columns=["Time"])

prices = pd.merge(price_series, price_data[["DateTime","Value"]], left_on="Time", right_on="DateTime", how="left").rename(columns={"Value":"spot_cost"})
prices = prices[pd.notnull(prices["DateTime"])]
prices = prices.drop(columns=["Time"])


prices["Time_String"] = prices["DateTime"].dt.tz_localize(None).dt.strftime("%H:%M:%S").astype(str)
prices["Quarter"] = ["Q" + value for value in np.ceil(prices["DateTime"].dt.month/3).astype(int).astype(str).to_list()]
prices = prices.iloc[:,[0,2,3,1]]




# ====== DYNAMIC NETWORK CHARGE PRICES =====

# read in network tariffs
network_charges_hours = pd.read_excel(r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v3.xlsx", sheet_name=0)
network_charges_euro = pd.read_excel(r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v3.xlsx", sheet_name=1)


network_levels_HSN = network_charges_hours[network_charges_hours["Netzbetreiber"].isin(netzbetreiber)].set_index("Netzbetreiber").rename_axis(None).iloc[:,0:96]
network_levels_S = network_levels_HSN.replace("H","S").replace("N","S")
array_quarters = network_charges_hours[network_charges_hours["Netzbetreiber"].isin(netzbetreiber)].set_index("Netzbetreiber").rename_axis(None).iloc[:,97:102].transpose()


HSN_table_stacked = pd.DataFrame() # create empty pd
quarter_title = ["Q1","Q2","Q3","Q4"]

for ct_q in quarter_title: # loop over all quarters
    
    HSN_table_temp = [network_levels_HSN.loc[nct] if array_quarters.loc[ct_q][nct] == 1 else network_levels_S.loc[nct] for nct in netzbetreiber] # get S or HSN
    HSN_table_temp = pd.DataFrame(HSN_table_temp).transpose()
    HSN_table_temp["Quarter"] = ct_q
    HSN_table_temp = pd.concat([HSN_table_temp["Quarter"], HSN_table_temp.iloc[:,0:-1]], axis = 1)
    HSN_table_temp = HSN_table_temp.reset_index()
    HSN_table_stacked = pd.concat([HSN_table_stacked, HSN_table_temp], axis = 0)


HSN_table_stacked = HSN_table_stacked.rename(columns={"index":"Time_String"})
HSN_table_stacked["Time_String"] = HSN_table_stacked["Time_String"].astype(str) # prepare dtypes for join/merge
HSN_table_stacked["Quarter"] = HSN_table_stacked["Quarter"].astype(str)


charges_HSN = network_charges_euro[network_charges_euro["Netzentgelte in Brutto"].isin(netzbetreiber)][["Netzentgelte in Brutto", "AP_HT_ct/kWh", "AP_ST_ct/kWh", "AP_NT_ct/kWh"]].transpose()
charges_HSN.columns = charges_HSN.iloc[0]
charges_HSN = charges_HSN.drop(charges_HSN.index[0])
charges_HSN.index = charges_HSN.index.str.replace("AP_","").str.replace("T_ct/kWh","")


# fill ToU segments with network charge numbers
# 1) create array of correct dimensions
network_charges_red_stacked = HSN_table_stacked.copy() # reduced network charges
network_charges_reg_stacked = HSN_table_stacked.copy() # regular

# 2) overwrite with numbers
for ct_n in network_charges_red_stacked.columns[2:]: # loop over all network operator columns
    network_charges_red_stacked[ct_n] = HSN_table_stacked[ct_n].map(charges_HSN[ct_n])
   
for ct_n in network_charges_reg_stacked.columns[2:]: # loop over all network operator columns
   network_charges_reg_stacked[ct_n] = charges_HSN[ct_n]["S"]


# create series for whole year
prices_red_seperate = pd.merge(prices, network_charges_red_stacked, on=["Time_String", "Quarter"], how="left", suffixes=["_x",""])
prices_reg_seperate = pd.merge(prices, network_charges_reg_stacked, on=["Time_String", "Quarter"], how="left", suffixes=["_x",""])
# sum up price components to obtain one price timeseries
prices_red = pd.concat([prices_red_seperate.iloc[:,1:3], prices_red_seperate.iloc[:,4:].add(prices["spot_cost"].reset_index(drop=True), axis='rows')], axis=1)
prices_reg = pd.concat([prices_reg_seperate.iloc[:,1:3], prices_reg_seperate.iloc[:,4:].add(prices["spot_cost"].reset_index(drop=True), axis='rows')], axis=1)
   
if (False):
    plt.figure()
    pd.DataFrame({"reduced":prices_red.iloc[:,2], "regular":prices_reg.iloc[:,2]}).plot(drawstyle="steps-post", style=["-","--"], color=["r","k"], ylabel="ct/kWh", xlabel="time")


  
if param_price == "red":
    prices_xr = xr.DataArray(prices_red.iloc[:,2:], dims=['t','r'])
elif param_price == "reg":
    prices_xr = xr.DataArray(prices_reg.iloc[:,2:], dims=['t','r'])



# ====== load temperature ======

# use dummy temperature from nodal Flex 
temperature = pd.read_csv(r"Z:\10_Paper\13_Alleinautorenpaper\temperature_nodalFlex.csv", encoding='utf-8')
temperature_cut = temperature.iloc[0,0:len(prices)].to_numpy()
alpha = 0.0025  # W/(m^2*K)
surface = 200 #m^2
limit_temp = 15 # °C

heat_demand = alpha * surface * np.maximum(limit_temp-temperature_cut, 0) 
heat_demand_xr = xr.DataArray(heat_demand, dims='t')


if (False):
    plt.plot(heat_demand)




# ====== Heat pump parameters =====
e_max = 20  # kWh
p_hp = 4 # kW
cop = 3 # [-]
timesteplength = 1 # h

# ===== optimization model =====


# Erstelle ein "Linopy"-Optimierungsmodell
m = Model()


# Definiere Sets
set_time = pd.Index(range(0,len(prices)), name="t")
set_dso = pd.Index(netzbetreiber, name="r")
set_setup = pd.Index(["reg", "red"], name="s")


# Endogene Variablen
P_HP = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP', lower=0) # Leistung Wärmepumpe
P_HP_slack = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HP_slack', lower=0) # Leistung Wärmepumpe

P_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='P_HStor') # EINspeicherung SPeicher
E_HStor = m.add_variables(coords=[set_time,set_dso, set_setup], name='E_HStor', lower=0) # EINspeicherung SPeicher

# Nebenbedingugen
cons_hp_min = m.add_constraints(P_HP - P_HP_slack >= 0, name='power_hp_min')
cons_hp_max = m.add_constraints(P_HP - P_HP_slack <= p_hp, name='power_hp_max')
cons_stor_max = m.add_constraints(E_HStor <= e_max, name='max_soc')
cons_stor_exchange = m.add_constraints(E_HStor.isel(t=range(1,len(set_time)-1)) == E_HStor.isel(t=range(0,len(set_time)-2)) + P_HStor.isel(t=range(0,len(set_time)-2)) * timesteplength, name='system_balance')
cons_stor_fix_last = m.add_constraints(P_HStor.isel(t=-1) == 0, name='p_hstor_fix_last_entry')
cons_stor_circle = m.add_constraints(E_HStor.isel(t=0) == E_HStor.isel(t=-2), name='power_hp_circle')
cons_demand_balance = m.add_constraints(cop * P_HP - P_HStor - heat_demand_xr == 0, name='demand_balance')

cons_stor_exchange_max = m.add_constraints(P_HStor <= 1, name='cons_stor_exchange_max')
cons_stor_exchange_min = m.add_constraints(P_HStor >= -1, name='cons_stor_exchange_min')



# zu minimierende Zielfunktion
obj = (prices_xr * P_HP).sum() + (penalty * P_HP_slack).sum()
m.add_objective(obj)


# Modell ansehen
print(m)


# Rechne das Optimierungsproblem
m.solve('gurobi')

#labels = m.compute_infeasibilities()
#m.print_infeasibilities()

# Ergebnis für Variable p ausgeben

result_p = P_HP.solution.to_dataframe().unstack()
slack_p = P_HP_slack.solution.to_dataframe().unstack()
print("======== SLACKS =========")
print(slack_p.sum())
print("=========================")


result_E_HStor = E_HStor.solution.to_dataframe()
result_P_HP =  P_HP.solution.to_dataframe()


result_p_ohne_slack = (P_HP.solution.to_dataframe().unstack() - penalty * slack_p).sum()

result_E_HStor = E_HStor.solution.to_dataframe().unstack()






# plots
if (False):
    result_P_HP.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="power in kW", xlabel="time")


if (False):
    result_E_HStor.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="energy in kWh", xlabel="time")


result_cost = (prices_xr * result_p).sum('t').to_pandas()
einsparung = (result_cost["reg"] - result_cost["red"]) / result_cost["reg"] * 100
print("Einsparung in Prozent: ", str(einsparung))