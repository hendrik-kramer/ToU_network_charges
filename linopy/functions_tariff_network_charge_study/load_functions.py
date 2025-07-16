
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
import os
import glob
import re
#from datetime import date, timedelta
import warnings
import datetime



def load_timesteps(input_year):
    
    parameter_year = input_year
    # get relevant weeks to compute
    str_start_check = pd.Series(pd.date_range(str(parameter_year-1) + "-12-25 00:00:00", periods=14, freq="d")) #.astype('datetime64[s]') 
    timestep_start = str_start_check[str_start_check.dt.isocalendar().week == 1].iloc[0] - pd.Timedelta(hours=1) # minus 1 for UTC - 1
    str_end_check = pd.Series(pd.date_range(str(parameter_year) + "-12-25 23:45:00", periods=14, freq="d")) #.astype('datetime64[s]')
    timestep_end = str_end_check[str_end_check.dt.isocalendar().week != 1].iloc[-1] - pd.Timedelta(hours=1)  # minus 1 for UTC - 1
    
    timesteps_string = pd.date_range(start=timestep_start, end=timestep_end, freq='15min').strftime('%Y-%m-%d %H:%M:%S')
    timesteps_df_col = pd.DataFrame({"DateTime":timesteps_string})
    timesteps_series = pd.Series(timesteps_string)
    timesteps = pd.DataFrame()
    #timesteps["DateTime"] = pd.to_datetime(timesteps_series).astype('datetime64[s]').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    timesteps["DateTime"] = pd.to_datetime(timesteps_series).dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    timesteps["Quarter"] = "Q" + np.ceil(timesteps["DateTime"].dt.month/3).astype(int).astype(str)
    timesteps.loc[1:1000, "Quarter"] = "Q1" # overwrite if start is in last year
    timesteps.loc[len(timesteps)-1000:, "Quarter"] = "Q4" # overwrite if end is in new year    
    timesteps["TimeString"] = timesteps["DateTime"].dt.strftime('%H:%M:%S')
    timesteps["DateString"] = timesteps["DateTime"].dt.strftime('%Y-%m-%d')
    
    epoch_time = datetime.datetime(1970, 1, 1)
    timesteps_utc = timesteps["DateTime"].dt.tz_convert("UTC").dt.tz_convert(None)
    timesteps["seconds_since_1970_in_utc"] = (timesteps_utc - epoch_time).dt.total_seconds() 
    
    return timesteps


def load_spot_prices(input_year, input_folderpath, str_auction, timesteps):

    # read in time data 
    if "da_auction_hourly_12_uhr" in str_auction:
        raw_price_data = pd.read_csv(input_folderpath + "da_auktion_12_uhr_hourly\energy-charts_DAM_hourly_" + str(input_year) + "_mit_Raendern.csv", skiprows=1)
        raw_price_data = raw_price_data.rename(columns={"Unnamed: 0":"Time_utc", "Preis (EUR/MWh, EUR/tCO2)":"Value"})
        raw_price_data["Time_utc"] = pd.to_datetime(raw_price_data['Time_utc'], utc=True)

        # interpolate hourly data to quarter data
        resample_method = "interpolate" # "stairs", "linear"
        if "stairs" in str_auction:
            raw_price_data = raw_price_data.resample("15min", on="Time_utc").mean().fillna(method='ffill')
        else:
            raw_price_data = raw_price_data.resample("15min", on="Time_utc").mean().shift(2).interpolate(method="linear")  # shift value to half hour value
                
        raw_price_data['Time_DE'] = raw_price_data.index.tz_convert('Europe/Berlin')
        raw_price_data['iso_year'] = raw_price_data['Time_DE'].dt.isocalendar().year
        raw_price_data['iso_week'] = raw_price_data['Time_DE'].dt.isocalendar().week
        raw_price_data["Value"] = raw_price_data["Value"]/10 # €/MWh --> ct/kWh
    
        #price_data = raw_price_data[(raw_price_data['iso_year']==input_year)]
        price_data = raw_price_data
        price_data.loc[:,"DateTime"] = pd.to_datetime(price_data["Time_DE"])
        
        price_data["Quarter"] = "Q" + np.ceil(pd.to_datetime(price_data["DateTime"]).dt.month/3).astype(int).astype(str)
        price_data = price_data.rename(columns={"Value":"spot_cost"})
        price_data["Time_String"] = pd.to_datetime(price_data["DateTime"]).dt.tz_localize(None).dt.strftime("%H:%M:%S").astype(str)
            
        price_data = price_data.rename(columns={"DateTime":"t"}).set_index("t", drop=True)
        price_data_xr = xr.DataArray(price_data["spot_cost"])
        prices_xr = price_data_xr.astype(float)
    
    elif str_auction == "da_auction_quarterly_12_uhr":
        
        #print("load 12h auction data")
        raw_price_data = pd.read_csv(input_folderpath + "da_auktion_12_uhr_quarterly\DayAheadPrices_12_1_D_2019_2024_hourly_quarterly.csv")
        #print(raw_price_data)
        # convert utc time to local time
        raw_price_data['Time_utc'] = pd.to_datetime(raw_price_data['DateTime'], format='%Y-%m-%d %H:%M:%S', utc=True)
        raw_price_data['Time_DE'] = raw_price_data['Time_utc'].dt.tz_convert('Europe/Berlin')
        raw_price_data['iso_year'] = raw_price_data['Time_DE'].dt.isocalendar().year
        raw_price_data['iso_week'] = raw_price_data['Time_DE'].dt.isocalendar().week
        raw_price_data["Value"] = raw_price_data["Value"]/10 # €/MWh --> ct/kWh
        
        price_data = raw_price_data[(raw_price_data['iso_year']==input_year) & (raw_price_data['ResolutionCode']=="PT15M")]
        price_data.loc[:,"DateTime"] = pd.to_datetime(price_data["Time_DE"])
        
        price_data["Quarter"] = "Q" + np.ceil(pd.to_datetime(price_data["DateTime"]).dt.month/3).astype(int).astype(str)
        price_data = price_data.rename(columns={"Value":"spot_cost"})
        price_data["Time_String"] = pd.to_datetime(price_data["DateTime"]).dt.tz_localize(None).dt.strftime("%H:%M:%S").astype(str)
        
        #price_data = price_data[["DateTime", "spot_cost"]]
        #prices_xr = xr.DataArray(price_data, dims=['t','r'])
    
        price_data = price_data.rename(columns={"DateTime":"t"}).set_index("t", drop=True)
        price_data = xr.DataArray(price_data["spot_cost"])
        prices_xr = price_data.astype(float)
    
    else: # "id_auktion_15_uhr" 
        print("load 15h auction data")
        files = glob.glob(os.path.join(input_folderpath,str_auction, '*.csv'))
        print(files)
        all_files = pd.DataFrame()
        for ct_file in files:
            
            ct_csv = pd.read_csv(ct_file, skiprows=1)
            warnings.filterwarnings("ignore")
            ct_csv = ct_csv.set_index(pd.to_datetime(ct_csv["Delivery day"], format="%d/%m/%Y")).drop(columns="Delivery day")
            warnings.filterwarnings("default")
            ct_csv = ct_csv.drop(ct_csv.columns[ct_csv.columns.str.contains("Hour") == False], axis=1)
            ct_csv.columns = ct_csv.columns.str.replace("Hour ","").str.replace("A","").str.replace("Q1","00:00").str.replace("Q2","15:00").str.replace("Q3","30:00").str.replace("Q4","45:00").str.replace(" ",":")
            col_hours = [re.split(r'[:B]+', ct_col)[0] for ct_col in ct_csv.columns]
            col_b = [re.findall(r'B', ct_col) for ct_col in ct_csv.columns]
            col_b = [ct_col[0] if len(ct_col) > 0 else "" for ct_col in col_b]
            col_rest = ct_csv.columns.to_series().str.split(":",n=1).str[1].to_list()
            ct_csv.columns = pd.Series([str(int(ct_col)-1) for ct_col in col_hours]).astype(str) + col_b + [":"]*len(col_rest) + col_rest
            ct_csv_stack = ct_csv.stack(level=-1, dropna=True).reset_index().rename(columns={"level_1":"time", 0:"spot_price"})
            ct_csv_stack["helper_hour"] = ct_csv_stack["time"].str.split(':').str[0].str.replace("B",".5").astype(float)
            ct_csv_stack = ct_csv_stack.sort_values(["Delivery day", "helper_hour"])
            
            all_files = pd.concat([all_files, ct_csv_stack], axis=0)
            # cannot be sorted given 2x hour B
        all_files = all_files.reset_index()
        
        # create generic timeseries, based on utc
        first_day = timesteps["DateTime"].iloc[0].strftime('%Y-%m-%d')
        first_time = str(timesteps["DateTime"].iloc[0].hour) + ":" + timesteps["DateTime"].iloc[0].strftime('%M:%S')
        idx_first_datetime = all_files[(all_files["Delivery day"] == first_day) & (all_files["time"] == first_time)].index[0]
        prices_15h_year = all_files.loc[idx_first_datetime:idx_first_datetime+len(timesteps)-1,"spot_price"]
        price_data = pd.concat([timesteps["DateTime"].reset_index(drop=True), prices_15h_year.reset_index(drop=True)], axis = 1)
        price_data["DateTime"]= pd.to_datetime(price_data["DateTime"])
        price_data = price_data.rename(columns={"DateTime":"t"})
        
        price_data["spot_price"] = price_data["spot_price"]/10 # €/MWh -> ct/kWh
        price_data = price_data.set_index("t", drop=True)
        price_data_series = price_data["spot_price"]
        
        price_data_xr = xr.DataArray(price_data_series)
        prices_xr = price_data_xr.astype(float)
    
    return prices_xr


def get_annual_static_tariff_prices(spot_prices_xr):
    
    tarrif_prices_xr = spot_prices_xr
    tarrif_prices_xr = np.round(spot_prices_xr.mean().item(), decimals=3)
        
    return tarrif_prices_xr
    


def load_network_charges(input_filepath, timesteps):

    network_charges_hours = pd.read_excel(input_filepath, sheet_name=0).head(100)
    network_charges_euro = pd.read_excel(input_filepath, sheet_name=1).head(100)
    
    network_levels_HSN = network_charges_hours.set_index("Netzbetreiber").rename_axis(None).iloc[:,0:96]
    network_levels_S = network_levels_HSN.replace("H","S").replace("N","S")
    array_quarters = network_charges_hours.set_index("Netzbetreiber").rename_axis(None).iloc[:,96:101].transpose()
      
    HSN_table_stacked = pd.DataFrame() # create empty pd
    quarter_title = ["Q1","Q2","Q3","Q4"]
    
    dsos = network_charges_hours["Netzbetreiber"].to_list()
    for ct_q in quarter_title: # loop over all quarters            
        HSN_table_temp = [network_levels_HSN.loc[nct] if array_quarters.loc[ct_q][nct] == 1 else network_levels_S.loc[nct] for nct in dsos] # get S or HSN
        HSN_table_temp = pd.DataFrame(HSN_table_temp).transpose()
        HSN_table_temp["Quarter"] = ct_q
        HSN_table_temp = pd.concat([HSN_table_temp["Quarter"], HSN_table_temp.iloc[:,0:-1]], axis = 1)
        HSN_table_temp = HSN_table_temp.reset_index()
        HSN_table_stacked = pd.concat([HSN_table_stacked, HSN_table_temp], axis = 0)
    
    
    HSN_table_stacked = HSN_table_stacked.rename(columns={"index":"TimeString"})
    HSN_table_stacked["TimeString"] = HSN_table_stacked["TimeString"].astype(str) # prepare dtypes for join/merge
    HSN_table_stacked["Quarter"] = HSN_table_stacked["Quarter"].astype(str)
    
    
    charges_HSN = network_charges_euro[network_charges_euro["Netzentgelte in Brutto"].isin(dsos)][["Netzentgelte in Brutto", "AP_HT_ct/kWh", "AP_ST_ct/kWh", "AP_NT_ct/kWh"]].transpose()
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
    
    # network charges for 4x (Q1, Q2, Q3, Q4) x 96 timesteps x DSOs
    network_charges_red = network_charges_red_stacked #.set_index("Time_String")
    network_charges_reg = network_charges_reg_stacked # .set_index("Time_String")

    #print(network_charges_red)

    if (False):
        plt.figure()
        pd.DataFrame({"reduced":network_charges_red, "regular":network_charges_reg}).plot(drawstyle="steps-post", style=["-","--"], color=["r","k"], ylabel="ct/kWh", xlabel="time")
    
    
    # repeat to get the full year's network charge timeseries
    timesteps = timesteps.drop("seconds_since_1970_in_utc", axis=1)
    network_charges_red_year = pd.merge(timesteps, network_charges_red, how="left", on=("TimeString", "Quarter")).set_index("DateTime", drop=True).drop(columns=["TimeString", "Quarter"])
    network_charges_reg_year = pd.merge(timesteps, network_charges_reg, how="left", on=("TimeString", "Quarter")).set_index("DateTime", drop=True).drop(columns=["TimeString", "Quarter"])
    network_charges_red_year = network_charges_red_year.drop(columns="DateString")
    network_charges_reg_year = network_charges_reg_year.drop(columns="DateString")

    #print(network_charges_red_year)
    
    network_charges_xr = xr.concat([xr.DataArray(network_charges_reg_year, dims=['t','r']).expand_dims("s",axis=2), xr.DataArray(network_charges_red_year, dims=['t','r']).expand_dims("s",axis=2)], dim="s")
    network_charges_xr['s'] = ['reg', 'red']
    
    
    
    # prepare secondary outpu no 1: amount of active quarters
    array_quarters_sum = array_quarters.loc[["Q1", "Q2", "Q3", "Q4"],:].transpose().sum(axis=1)
    xr_dso_quarters_sum = xr.DataArray(array_quarters_sum, dims="r")
    
    # prepare secondary output n0 2: ht length
    network_levels_H = (network_levels_HSN == "H").sum(axis=1)/4
    xr_ht_length = xr.DataArray(network_levels_H , dims="r" )
    
    network_levels_N = (network_levels_HSN == "N").sum(axis=1)/4
    xr_nt_length = xr.DataArray(network_levels_N , dims="r" )
    
    # prepare secondary output no 3: ht/st price factor
    xr_ht_charge = xr.DataArray( charges_HSN.loc["H",:] , dims="r")
    xr_st_charge = xr.DataArray( charges_HSN.loc["S",:] , dims="r")
    xr_nt_charge = xr.DataArray( charges_HSN.loc["N",:] , dims="r")
    
    
    return network_charges_xr.astype(float), xr_dso_quarters_sum, xr_ht_length, xr_nt_length, xr_ht_charge, xr_st_charge, xr_nt_charge


def load_emob(input_filepath_emob_demand, input_filepath_emob_state, timesteps):

    emob_demand = pd.read_csv(input_filepath_emob_demand, encoding='utf-8')
    emob_demand["date"] = pd.to_datetime(emob_demand["date"])
    first_datetime_timestep = timesteps["DateTime"].iloc[0].strftime('%Y-%m-%d %H:%M:%S')

    
    if emob_demand.iloc[1000].date.year != int(first_datetime_timestep[0:4]):
        offset = 365 * (int(first_datetime_timestep[0:4]) - emob_demand.iloc[1000].date.year)
        print("================================================================")
        print("!!! Emob timesteps do not match. Shift emob dates by days: " + str(offset) + ", i.e. weeksdays and daylight saving not correct !!!")
        emob_demand.date = emob_demand.date + pd.Timedelta(offset, unit="D")

    
    idx_first_datetime = emob_demand[emob_demand["date"].dt.strftime('%Y-%m-%d %H:%M:%S') == first_datetime_timestep].index[0]
    emob_year = emob_demand.loc[idx_first_datetime:idx_first_datetime+len(timesteps)-1]
    emob_year = emob_year.rename(columns={"date":"t"}).set_index("t", drop=True)
    time_offset = pd.to_datetime(emob_year.index) - datetime.timedelta(hours=1)
    emob_year.index = time_offset.tz_localize("utc").tz_convert("Europe/Berlin")
    emob_demand_xr = xr.DataArray(emob_year, dims=['t','v'])
    
    emob_demand = pd.read_csv(input_filepath_emob_state, encoding='utf-8')
    emob_demand["date"] = pd.to_datetime(emob_demand["date"])
    first_datetime_timestep = timesteps["DateTime"].iloc[0].strftime('%Y-%m-%d %H:%M:%S')

    
    if emob_demand.iloc[1000].date.year != int(first_datetime_timestep[0:4]):
        offset = 365 * (int(first_datetime_timestep[0:4]) - emob_demand.iloc[1000].date.year)
        print("!!! Emob timesteps do not match. Shift emob dates by days: " + str(offset) + ", i.e. weeksdays and Daylight Saving not correct !!!")
        print("================================================================")
        emob_demand.date = emob_demand.date + pd.Timedelta(offset, unit="D")
    
    idx_first_datetime = emob_demand[emob_demand["date"].dt.strftime('%Y-%m-%d %H:%M:%S') == first_datetime_timestep].index[0]
    emob_year = emob_demand.loc[idx_first_datetime:idx_first_datetime+len(timesteps)-1]
    emob_year = emob_year.rename(columns={"date":"t"}).set_index("t", drop=True)
    time_offset = pd.to_datetime(emob_year.index) - datetime.timedelta(hours=1)
    emob_year.index = time_offset.tz_localize("utc").tz_convert("Europe/Berlin")
    emob_state_xr = xr.DataArray(emob_year, dims=['t','v'])
    
    return emob_demand_xr, emob_state_xr


def deduce_arrival_departure_times(emob_demand_xr, emob_state_xr, timesteps, shift_timesteps_int):

    amt_timesteps = len(timesteps)
    
    ev_demand = emob_demand_xr.to_pandas()
    ev_home = emob_state_xr=="home"
    ev_driving = emob_state_xr=="driving"
    
        
    # most frequent departure
    emob_drives_now = (ev_driving & ev_home.shift(t=+1,fill_value=True)).to_pandas()
    emob_drives_now_daily_count = emob_drives_now.groupby(by=emob_drives_now.index.time).sum()
    emob_drives_now_daily_count_morning = emob_drives_now_daily_count[emob_drives_now_daily_count.index < datetime.time(hour=15)]
    emob_drives_now_fequent_hour = emob_drives_now_daily_count_morning.idxmax()

    # most frequent arrival
    emob_home_now = (ev_home & ev_driving.shift(t=+1,fill_value=True)).to_pandas()
    emob_home_now_daily_count = emob_home_now.groupby(by=emob_home_now.index.time).sum()
    emob_home_now_daily_count_evening = emob_home_now_daily_count[emob_home_now_daily_count.index >= datetime.time(hour=15)]
    emob_home_now_fequent_hour = emob_home_now_daily_count_evening.idxmax()

    times_arrival = emob_home_now_fequent_hour
    times_departure = emob_drives_now_fequent_hour


    #pandas_false = pd.DataFrame( np.full((amt_timesteps,len(emob_demand_xr.v)),False), columns=emob_demand_xr.v, index=ev_demand.index)
    
    #emob_departure_df = pandas_false
    #emob_arrival_df = pandas_false
    
    #for ct_col in emob_departure_df:
    #    ct_ev_pref_hour_departure = emob_drives_now_fequent_hour.iloc[[ct_col]].iloc[0]
    #    emob_departure_df.loc[emob_departure_df.index.time == ct_ev_pref_hour_departure, ct_col] = True

    #    ct_ev_pref_hour_arrival = emob_home_now_fequent_hour.iloc[[ct_col]].iloc[0]
    #    emob_departure_df.loc[emob_departure_df.index.time == ct_ev_pref_hour_arrival, ct_col] = True

    #emob_departure_xr = xr.DataArray(emob_departure_df, dims=["t","v"])
    #emob_arrival_xr = xr.DataArray(emob_arrival_df, dims=["t","v"])

    unique_timesteps = timesteps["DateTime"].dt.time.unique()
    dict_id_timesteps = {}
    
    for ct_ts in unique_timesteps:
        dict_id_timesteps[ct_ts] = timesteps.index[ct_ts == timesteps["DateTime"].dt.time] + shift_timesteps_int


    return times_arrival, times_departure, dict_id_timesteps



def load_temperature(input_filepath_temperature, timesteps):
    temperature = pd.read_csv(input_filepath_temperature, header=None, encoding='utf-8').transpose()
    parameter_year = timesteps.iloc[10]["DateTime"].year
    temperature.index = pd.date_range(start=str(parameter_year)+"-01-01 00:00:00", end=str(parameter_year)+"-12-31 23:00:00", freq="1h")
    temperature = temperature.resample("15min").interpolate(method="spline", order=3, s=0)
    temperature = pd.concat([temperature, pd.DataFrame(temperature.iloc[[-1, -2, -3]], index=pd.date_range(start=str(parameter_year)+"-12-31 23:15:00", end=str(parameter_year)+"-12-31 23:45:00", freq="15min"))], axis=0) # add missing last 3 quarters
    temperature = temperature.fillna(method="ffill")
    if len(temperature) > len(timesteps): # make length match and change index
        temperature = temperature.iloc[:len(timesteps)].set_index(pd.to_datetime(timesteps.DateTime))
    else:
        print("Error --> timesteps > temperature, must copy last week, todo")
    temperature_xr = xr.DataArray(temperature, dims=['t','z'])
    return temperature_xr


def load_irradiance(input_folderpath, parameter_file_capacities, parameter_file_fulloadhours, timesteps):
    
    hochrechnung = pd.read_csv(input_folderpath, sep=";", decimal=',', na_values ={"N.A", "N.A."}).dropna(axis=0) # in MW, in UTC
    hochrechnung["Zeit"] = pd.to_datetime(hochrechnung["Zeit"], utc=True).dt.tz_convert("Europe/Berlin")
    hochrechnung = hochrechnung.set_index("Zeit", drop=True)
        
    # heal missing data
    hochrechnung_timesteps = pd.DataFrame(index=pd.date_range(start=hochrechnung.index[0], end=hochrechnung.index[-1], freq="15min"))
    
    hochrechnung = pd.merge(hochrechnung_timesteps, hochrechnung, left_index=True, right_index=True, how="left")
    hochrechnung = hochrechnung.fillna(0)
    
    # Missing Feb 2 -- Feb 5
    tmp_vals = np.array(hochrechnung.loc[(hochrechnung.index.date>=datetime.date(2024, 2, 7)) & (hochrechnung.index.date<=datetime.date(2024, 2, 10)), ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']])
    hochrechnung.loc[(hochrechnung.index.date>=datetime.date(2024, 2, 2)) & (hochrechnung.index.date<=datetime.date(2024, 2, 5)),['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = tmp_vals

    # Missing Mar 31
    tmp_vals = np.array(hochrechnung.loc[hochrechnung.index.date==datetime.date(2024, 3, 30), ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']])
    hochrechnung.loc[(hochrechnung.index.date==datetime.date(2024, 3, 31)) ,['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = tmp_vals[0:92,:]

    # Missing Mar 3 -- Mar 5
    tmp_vals = np.array(hochrechnung.loc[(hochrechnung.index.date>=datetime.date(2024, 4, 6)) & (hochrechnung.index.date<=datetime.date(2024, 4, 8)), ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']])
    hochrechnung.loc[(hochrechnung.index.date>=datetime.date(2024, 4, 3)) & (hochrechnung.index.date<=datetime.date(2024, 4, 5)),['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = tmp_vals

    #hochrechnung.loc[hochrechnung.index.date==datetime.date(2024, 3, 31), ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = hochrechnung.loc[hochrechnung.index.date==datetime.date(2024, 3, 30), ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] 
    # hochrechnung_year = hochrechnung.loc[hochrechnung.index.year == 2024]

    #hochrechnung.loc[:,['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = hochrechnung_splined.loc[:,['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']]
   
    #capacity = pd.read_csv(parameter_file_capacities, sep=";")[0:4].set_index("capacity_MWp").transpose()
    
    # normalize each timeseries per year to match in sum 1
    for ct_year in hochrechnung.index.year.unique():
        
        # annual sum
        sum_annual_val = hochrechnung.loc[hochrechnung.index.year == ct_year, ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']].sum()
        
        if ct_year == 2024:
            sum_annual_val_temp = hochrechnung.loc[hochrechnung.index.year == 2024, ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']].sum()
        elif ct_year == 2025:
            sum_annual_val = sum_annual_val_temp

        
        # normalize
        hochrechnung.loc[hochrechnung.index.year == ct_year, ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']] = \
            hochrechnung.loc[hochrechnung.index.year == ct_year, ['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']].div(sum_annual_val, axis=1)
            

    hochrechnung_year_timesteps = pd.DataFrame(timesteps.DateTime).set_index("DateTime", drop=True)
    hochrechnung_year = hochrechnung_year_timesteps.merge(hochrechnung[['50Hertz', 'Amprion', 'TenneT TSO', 'TransnetBW']], left_index=True, right_index=True, how="left")

    hochrechnung_year["meanTSO"] = hochrechnung_year.mean(axis=1) # still normlized to one

    # annual timeseries to reach fullloadhours of MiFri with one unit of power
    # Multiplicatio by 4 to account for quarter hours --> hours
    fullloadhours = pd.read_csv(parameter_file_fulloadhours, sep=";", index_col="Jahr").loc[hochrechnung_year.index[10].year,"Volllaststunden"]
    hochrechnung_year = fullloadhours * hochrechnung_year 

    hochrechnung_year["meanTSO"] = hochrechnung_year["meanTSO"]/hochrechnung_year["meanTSO"].sum() * fullloadhours # specifically rescale mean timeseries again to get rid of offset of end/beginning neighboring year's dates

    irradiance_xr = xr.DataArray(hochrechnung_year, dims=["t","a"])
    
    return irradiance_xr



def load_irradiance_dwd(input_folderpath, timesteps): # noch nicht ganz korrekt, Zickzack in Zeitreihen, wohl wegen 10-15min rescaling
    
    # Die Messungen sind vor dem Jahr 2000 einem Zeitstempel in MEZ und ab dem Jahr 2000 einem Zeitstempel in UTC zugeordnet.
    files_old = sorted(glob.glob(os.path.join(input_folderpath, 'produkt_zehn_min_sd_20100101_*.txt')))
    files_new = sorted(glob.glob(os.path.join(input_folderpath, 'produkt_zehn_min_sd_20200101_*.txt')))

    irradiance_xr = xr.DataArray() 

    for ct_file in range(0,len(files_old)):

        irradiance_old = pd.read_csv(files_old[ct_file], sep=";", na_values=-999).rename(columns={'  QN':'QN'})
        irradiance_old["MESS_DATUM"] = pd.to_datetime(irradiance_old["MESS_DATUM"], format="%Y%m%d%H%M") - timedelta(minutes=10) - timedelta(hours=1) # the irradiation sum of the 10 minutes is shifted to the the beginning of the time period
        irradiance_old = irradiance_old.set_index("MESS_DATUM", drop=True).tz_localize("utc").tz_convert("Europe/Berlin")
        
        irradiance_new = pd.read_csv(files_new[ct_file], sep=";", na_values=-999).rename(columns={'  QN':'QN'})
        irradiance_new["MESS_DATUM"] = pd.to_datetime(irradiance_new["MESS_DATUM"], format="%Y%m%d%H%M") - timedelta(minutes=10) # the irradiation sum of the 10 minutes is shifted to the the beginning of the time period
        irradiance_new = irradiance_new.set_index("MESS_DATUM", drop=True).tz_localize("UTC").tz_convert("Europe/Berlin")
        
        irradiance = pd.concat([irradiance_old, irradiance_new], axis=0)
        station_id = irradiance["STATIONS_ID"].iloc[0]
        irradiance = irradiance[["GS_10"]]
        
        irradiance_year = irradiance[(irradiance.index >= timesteps.iloc[0]["DateTime"]) & (irradiance.index <= timesteps.iloc[-1]["DateTime"] + timedelta(minutes=5) )]
        irradiance_year["GS_10"] = irradiance_year["GS_10"].interpolate("spline", order=3, s=0)
        irradiance_year = irradiance_year.resample("15min").sum()
        irradiance_year = irradiance_year.rename(columns={"GS_10":"i"})
        irradiance_year.index.names = [None]
    
        #irradiance_test = irradiance_year
        #irradiance_test["hour_decimal"] = irradiance_test.index.hour + irradiance_test.index.minute/60
        #irradiance_test_summer = irradiance_test[(irradiance_test.index.month > 4) & (irradiance_test.index.month < 9)]
        #irradiance_test_summer_group = irradiance_test_summer["GS_10"].groupby(irradiance_test_summer["hour_decimal"]).mean().reset_index()
        
        #plt.figure()
        #plt.bar(irradiance_test_summer_group.hour_decimal, irradiance_test_summer_group.GS_10)
 
        irradiance_xr_ct = xr.DataArray(irradiance_year, dims=("t","a"))
        #print(irradiance_xr_ct)
        if ct_file > 0:
            irradiance_xr = xr.concat([irradiance_xr, irradiance_xr_ct], dim="a")
        else:
            irradiance_xr = irradiance_xr_ct
            
    return irradiance_xr