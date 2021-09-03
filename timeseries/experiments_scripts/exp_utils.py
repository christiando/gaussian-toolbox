# data loaders here
from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries
import pandas as pd


from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../timeseries/')
sys.path.append('../src/')
import timeseries_data


def load_sunspots(ts=False, train_ratio=0.5):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    '''
    df = pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv')
    ts_df = TimeSeries.from_dataframe(df=df, 
                                      time_col='Month', 
                                      freq='MS')
    if not ts:
            df_dim = 1
            data = ts_df.all_values().squeeze()
            x_al = data.copy().reshape(-1, df_dim)
    
            train_length = int(x_al.shape[0] * train_ratio)
            val_length = int(0.1 * train_length)
            train_end = train_length-val_length
    
            x_tr = x_al[:train_end]
            x_va = x_al[train_end:train_end + val_length]
            x_te = x_al[train_end + val_length:]
    
            s_tr_x = StandardScaler().fit(x_tr)

            x_tr = s_tr_x.transform(x_tr)
            x_va = s_tr_x.transform(x_va)
            x_te = s_tr_x.transform(x_te)
    
            return x_tr, x_va, x_te, s_tr_x
    else:
        
        train_length =  int(len(ts_df) * train_ratio)
        val_length = int(0.1 * train_length)
        
        train_sp, x_te = ts_df.split_after(int(len(ts_df) * train_ratio))
        x_tr, x_va = train_sp.split_after(int(len(train_sp) * 0.9))

        s_tr_x = Scaler()
        x_tr = s_tr_x.fit_transform(x_tr)
        x_va = s_tr_x.transform(x_va)
        x_te = s_tr_x.transform(x_te)
    
        return x_tr, x_va, x_te, s_tr_x
    
    
def load_energy(ts=False, train_ratio=0.5):
    '''
    Loads the energy dataset for Germany, 4D time series
    as_ndarray: set to True if SSM models as predictors
    '''
    de_energy_df = pd.read_csv('../../data/opsd/time_series_60min_singleindex_filtered.csv')
    de_energy_df['utc_timestamp'] = pd.to_datetime(de_energy_df['utc_timestamp'].astype(str).apply(lambda x: x[:10]))
    de_energy_df = de_energy_df.groupby('utc_timestamp').mean()
    de_weather_df = pd.read_csv('../../data/opsd/weather_data_filtered.csv')
    de_weather_df['utc_timestamp'] = pd.to_datetime(de_weather_df['utc_timestamp'].astype(str).apply(lambda x: x[:10]))
    de_weather_df = de_weather_df.groupby('utc_timestamp').mean()
    
    full_df = pd.concat([de_energy_df[['DE_solar_generation_actual']], 
                             de_weather_df[['DE_temperature',
                                            'DE_radiation_direct_horizontal',
                                            'DE_radiation_diffuse_horizontal']]],
                        axis=1)
    full_df = full_df.fillna(method='backfill')

    ts_df = TimeSeries.from_dataframe(df=full_df.reset_index(), 
                                  time_col='utc_timestamp')
    if not ts:
            df_dim = 4
            data = ts_df.all_values().squeeze()
            x_al = data.copy().reshape(-1, df_dim)
    
            train_length = int(x_al.shape[0] * train_ratio)
            val_length = int(0.1 * train_length)
            train_end = train_length-val_length
    
            x_tr = x_al[:train_end]
            x_va = x_al[train_end:train_end + val_length]
            x_te = x_al[train_end + val_length:]
    
            s_tr_x = StandardScaler().fit(x_tr)

            x_tr = s_tr_x.transform(x_tr)
            x_va = s_tr_x.transform(x_va)
            x_te = s_tr_x.transform(x_te)
    
            return x_tr, x_va, x_te[:-1], s_tr_x
    else:
        
        train_length =  int(len(ts_df) * train_ratio)
        val_length = int(0.1 * train_length)
        
        train_sp, x_te = ts_df.split_after(int(len(ts_df) * train_ratio))
        x_tr, x_va = train_sp.split_after(int(len(train_sp) * 0.9))

        s_tr_x = Scaler()
        x_tr = s_tr_x.fit_transform(x_tr)
        x_va = s_tr_x.transform(x_va)
        x_te = s_tr_x.transform(x_te)
    
        return x_tr, x_va, x_te[:-1], s_tr_x
    


def load_synthetic(ts=False, train_ratio=0.5):
    data_df = timeseries_data.load_synthetic_data()
    data_df.index.name = 'utc_timestamp'
    ts_df = TimeSeries.from_dataframe(df=data_df.reset_index(),
                                 time_col='utc_timestamp')
    if not ts:
            df_dim = 7
            data = ts_df.all_values().squeeze()
            x_al = data.copy().reshape(-1, df_dim)
    
            train_length = int(x_al.shape[0] * train_ratio)
            val_length = int(0.1 * train_length)
            train_end = train_length-val_length
    
            x_tr = x_al[:train_end]
            x_va = x_al[train_end:train_end + val_length]
            x_te = x_al[train_end + val_length:]
    
            s_tr_x = StandardScaler().fit(x_tr)

            x_tr = s_tr_x.transform(x_tr)
            x_va = s_tr_x.transform(x_va)
            x_te = s_tr_x.transform(x_te)
    
            return x_tr, x_va, x_te[:-1], s_tr_x
    else:
        
        train_length =  int(len(ts_df) * train_ratio)
        val_length = int(0.1 * train_length)
        
        train_sp, x_te = ts_df.split_after(int(len(ts_df) * train_ratio))
        x_tr, x_va = train_sp.split_after(int(len(train_sp) * 0.9))

        s_tr_x = Scaler()
        x_tr = s_tr_x.fit_transform(x_tr)
        x_va = s_tr_x.transform(x_va)
        x_te = s_tr_x.transform(x_te)
    
        return x_tr, x_va, x_te[:-1], s_tr_x
    



def load_airfoil(ts=False, train_ratio=0.5):
    data_df = timeseries_data.load_airfoil_data()
    data_df.index.name = 'timestamp'
    ts_df = TimeSeries.from_dataframe(df=data_df.reset_index(),
                                 time_col='timestamp')
    if not ts:
            df_dim = 11
            data = ts_df.all_values().squeeze()
            x_al = data.copy().reshape(-1, df_dim)
    
            train_length = int(x_al.shape[0] * train_ratio)
            val_length = int(0.1 * train_length)
            train_end = train_length-val_length
    
            x_tr = x_al[:train_end]
            x_va = x_al[train_end:train_end + val_length]
            x_te = x_al[train_end + val_length:]
    
            s_tr_x = StandardScaler().fit(x_tr)

            x_tr = s_tr_x.transform(x_tr)
            x_va = s_tr_x.transform(x_va)
            x_te = s_tr_x.transform(x_te)
    
            return x_tr, x_va, x_te[:-1], s_tr_x
    else:
        
        train_length =  int(len(ts_df) * train_ratio)
        val_length = int(0.1 * train_length)
        
        train_sp, x_te = ts_df.split_after(int(len(ts_df) * train_ratio))
        x_tr, x_va = train_sp.split_after(int(len(train_sp) * 0.9))

        s_tr_x = Scaler()
        x_tr = s_tr_x.fit_transform(x_tr)
        x_va = s_tr_x.transform(x_va)
        x_te = s_tr_x.transform(x_te)
    
        return x_tr, x_va, x_te[:-1], s_tr_x