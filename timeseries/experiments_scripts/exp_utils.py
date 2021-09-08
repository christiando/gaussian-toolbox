# data loaders here
from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../timeseries/')
sys.path.append('../src/')
#import timeseries_data


def softplus_np(x): 
    print(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus(x_):
    """
    Softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = [np.log(1 + np.exp(-np.abs(x_[0]))) + np.maximum(x_[0], 0)]
    for i in range(1, len(x_)):
        if x_[i] is not []:
            y_ = y_ + [np.log(1 + np.exp(-np.abs(x_[i]))) + np.maximum(x_[i], 0)]
    return y_

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
    
            x_al = softplus_np(x_al)
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
    full_df = softplus_np(full_df)
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
    data_df = softplus_np(data_df)
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
    data_df = softplus_np(data_df)
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
    
    
    
def load_sunspots_e1(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    
    experiment 2: fill in missing data
    '''
    df = np.asarray(pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv'))

    df_dim = 2
    x_al = df.copy().reshape(-1, df_dim)
    
    # scale the whole dataset at once since we will remove points from the whole range
    s_x = StandardScaler().fit(x_al[:, 1].reshape(-1, 1))
    x_al[:, 1] = s_x.transform(x_al[:, 1].reshape(-1, 1)).reshape(-1)
    x_al[:, 1] = softplus(np.asarray(x_al[:, 1], dtype=np.float32))

    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)

  
    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=['Month', 'Sunspots'])
        ts_tr = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_va, columns=['Month', 'Sunspots'])
        ts_va = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_te, columns=['Month', 'Sunspots'])
        ts_te = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')

        return ts_tr, ts_va, ts_te, s_x
    
    else:
        
        return x_tr, x_va, x_te, s_x
    
def load_sunspots_e2(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    
    experiment 2: fill in missing data
    '''
    df = np.asarray(pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv'))

    df_dim = 2
    x_al = df.copy().reshape(-1, df_dim)
    
    # scale the whole dataset at once since we will remove points from the whole range
    s_x = StandardScaler().fit(x_al[:, 1].reshape(-1, 1))
    x_al[:, 1] = s_x.transform(x_al[:, 1].reshape(-1, 1)).reshape(-1)
    x_al[:, 1] = softplus(np.asarray(x_al[:, 1], dtype=np.float32))

    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)

    df_tmp = pd.DataFrame(data = x_tr, columns=['Month', 'Sunspots'])
    ts_tr = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    df_tmp = pd.DataFrame(data = x_va, columns=['Month', 'Sunspots'])
    ts_va = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    df_tmp = pd.DataFrame(data = x_te, columns=['Month', 'Sunspots'])
    ts_te = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    ts_te_full = ts_te.copy()
    x_missing_arr = ts_te.values()
    x_missing_arr_t = ts_te._time_index._data
    to_del = np.random.rand(x_te.shape[0]) < delete_ratio
    x_missing_arr[to_del] = np.nan
    x_missing_arr_t[to_del] = np.nan

    if ts:
        
        return ts_tr, ts_va, ts_te_full, ts_te, s_x
    
    else:
        x_na = x_te.copy()
        x_na[to_del] = np.nan
        
        return x_tr, x_va, x_te, x_na, s_x
    

def load_sunspots_e1(ts=False, train_ratio=0.5):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    '''
    df = pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv')
    ts_df = TimeSeries.from_dataframe(df=df, 
                                      time_col='Month', 
                                      freq='MS')
    df_dim = 1
    data = ts_df.all_values().squeeze()
    df_dim = 2
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    x_al[:, 1] = softplus_np(np.asarray(x_al[:, 1], dtype=np.float32))
    
    x_al[:, 1] = np.asarray(x_al[:, 1], dtype=np.float16)
    train_length = int(x_al.shape[0] * train_ratio)
    val_length = int(0.1 * train_length)
    train_end = train_length-val_length
    
    x_tr = x_al[:train_end]
    x_va = x_al[train_end:train_end + val_length]
    x_te = x_al[train_end + val_length:]
    
    s_tr_x = StandardScaler().fit(x_tr[:, 1].reshape(-1, 1))

    x_tr[:, 1] = s_tr_x.transform(x_tr[:, 1].reshape(-1, 1)).reshape(-1)
    x_va[:, 1] = s_tr_x.transform(x_va[:, 1].reshape(-1, 1)).reshape(-1)
    x_te[:, 1] = s_tr_x.transform(x_te[:, 1].reshape(-1, 1)).reshape(-1)
    
    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=['Month', 'Sunspots'])
        ts_tr = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_va, columns=['Month', 'Sunspots'])
        ts_va = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_te, columns=['Month', 'Sunspots'])
        ts_te = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
    
        return ts_tr, ts_va, ts_te, s_tr_x
    
    else:
    
        return x_tr[:, 1].reshape(-1, 1), x_va, x_te, s_tr_x
    
    