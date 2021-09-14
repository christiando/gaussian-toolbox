# data loaders here
from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../../timeseries/')
sys.path.append('../../src/')
import timeseries_data


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

def inv_softplus(x):
    """
    Softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """

    y = np.empty(x.shape)
    y[x > 10] = x[x > 10]
    tmp = np.amax(np.stack([x[x <= 10], 1e-5 * np.ones(x[x<=10].shape)]), 0)
    tmp = np.asarray(tmp, dtype=np.float64)
    y[x <= 10] = np.log(np.exp(tmp) - 1)
    return y


def load_sunspots_e1(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    
    '''
    df = np.asarray(pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv'))

    df_dim = 2
    x_al = df.copy().reshape(-1, df_dim)
    #x_al[:, 1] = inv_softplus(np.asarray(x_al[:, 1], dtype=np.float32))
    x_al[:, 1] = inv_softplus(np.asarray(x_al[:, 1], dtype=np.float64))

    #x_al[:, 1] = np.asarray(x_al[:, 1], dtype=np.float64)
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)
    
    s_x = StandardScaler().fit(x_tr[:, 1].reshape(-1, 1))
    x_tr[:, 1] = s_x.transform(x_tr[:, 1].reshape(-1, 1)).reshape(-1)
    x_va[:, 1] = s_x.transform(x_va[:, 1].reshape(-1, 1)).reshape(-1)
    x_te[:, 1] = s_x.transform(x_te[:, 1].reshape(-1, 1)).reshape(-1)

  
    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=['Month', 'Sunspots'])
        ts_tr = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_va, columns=['Month', 'Sunspots'])
        ts_va = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
        df_tmp = pd.DataFrame(data = x_te, columns=['Month', 'Sunspots'])
        ts_te = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')

        return ts_tr, ts_va, ts_te, 0, s_x
    
    else:
        
        return np.asarray(x_tr[:, 1], dtype=np.float64).reshape(-1, 1),\
               np.asarray(x_va[:, 1], dtype=np.float64).reshape(-1, 1),\
               np.asarray(x_te[:, 1], dtype=np.float64).reshape(-1, 1),\
               0, s_x
    
def load_sunspots_e2(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    '''
    Loads the suspot dataset, 1D time series
    as_ndarray: set to True if SSM models as predictors
    
    experiment 2: fill in missing data
    '''
    df = np.asarray(pd.read_csv(filepath_or_buffer='../../data/sunspots/monthly-sunspots.csv'))

    df_dim = 2
    x_al = df.copy().reshape(-1, df_dim)
    x_al[:, 1] = inv_softplus(np.asarray(x_al[:, 1], dtype=np.float64))

    # scale the whole dataset at once since we will remove points from the whole range
    s_x = StandardScaler().fit(x_al[:, 1].reshape(-1, 1))
    x_al[:, 1] = s_x.transform(x_al[:, 1].reshape(-1, 1)).reshape(-1)
    
    
    # split in train test and prepare in ndarray and TimeSeries data format
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)

    df_tmp = pd.DataFrame(data = x_tr, columns=['Month', 'Sunspots'])
    ts_tr = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    df_tmp = pd.DataFrame(data = x_va, columns=['Month', 'Sunspots'])
    ts_va = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    df_tmp = pd.DataFrame(data = x_te, columns=['Month', 'Sunspots'])
    ts_te = TimeSeries.from_dataframe(df=df_tmp, time_col='Month')
        
    # remove random points from test dataset
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
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               np.asarray(x_na[:, 1:], dtype=np.float64), s_x
    

def load_energy_e1(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
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
    
    df = full_df.fillna(method='backfill')
    df.reset_index(inplace=True)
    col_names = df.columns
 
    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    x_al[:,  [1,3,4]] = np.asarray(x_al[:,  [1,3, 4]], dtype=np.float64)
    
    for c in [1, 3,4]:
        x_al[:,c] =  np.apply_along_axis(inv_softplus, 0, x_al[:,c])
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed, shuffle=False)

    s_x = StandardScaler().fit(x_tr[:, 1:])
    x_tr[:, 1:] = s_x.transform(x_tr[:, 1:])
    x_va[:, 1:] = s_x.transform(x_va[:, 1:])
    x_te[:, 1:] = s_x.transform(x_te[:, 1:])

    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
        ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')
        
        df_tmp = pd.DataFrame(data = x_va, columns=col_names)
        ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')
        
        df_tmp = pd.DataFrame(data = x_te, columns=col_names)
        ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')

        return ts_tr, ts_va, ts_te, 0, s_x
    
    else:
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               0, s_x
    
    
    
def load_energy_e2(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    '''
    Loads the energy dataset for Germany, 4D time series
    as_ndarray: set to True if SSM models as predictors
    
    fill-in missing data
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
    
    df = full_df.fillna(method='backfill')
    df.reset_index(inplace=True)
    col_names = df.columns
    

    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    x_al[:,  [1,3,4]] = np.asarray(x_al[:,  [1,3, 4]], dtype=np.float64)
    
    for c in [1, 3,4]:
        x_al[:,c] =  np.apply_along_axis(inv_softplus, 0, x_al[:,c])

    
    s_x = StandardScaler().fit(x_al[:, [1,3, 4]])
    x_al[:, [1,3, 4]] = s_x.transform(x_al[:, [1,3, 4]])
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)
    


    df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
    ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')
        
    df_tmp = pd.DataFrame(data = x_va, columns=col_names)
    ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')
        
    df_tmp = pd.DataFrame(data = x_te, columns=col_names)
    ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='utc_timestamp')

    # remove random points from test dataset
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
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               np.asarray(x_na[:, 1:], dtype=np.float64), s_x
    
    
   
def load_airfoil_e1(ts=False, train_ratio=0.5, seed=0):
    df = timeseries_data.load_airfoil_data()
    df.index.name = 'timestamp'
    
    df.reset_index(inplace=True)
    col_names = df.columns
 
    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
      
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed, shuffle=False)

    s_x = StandardScaler().fit(x_tr[:, 1:])
    x_tr[:, 1:] = s_x.transform(x_tr[:, 1:])
    x_va[:, 1:] = s_x.transform(x_va[:, 1:])
    x_te[:, 1:] = s_x.transform(x_te[:, 1:])

    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
        ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
        df_tmp = pd.DataFrame(data = x_va, columns=col_names)
        ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
        df_tmp = pd.DataFrame(data = x_te, columns=col_names)
        ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')

        return ts_tr, ts_va, ts_te, 0, s_x
    
    else:
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               0, s_x
    
    
def load_airfoil_e2(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    df = timeseries_data.load_airfoil_data()
    df.index.name = 'timestamp'
    
    df.reset_index(inplace=True)
    col_names = df.columns
 
    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    x_al[:, 1:] = np.asarray(x_al[:, 1:], dtype=np.float64)
    
    
    s_x = StandardScaler().fit(x_al[:, 1:])
    x_al[:, 1:] = s_x.transform(x_al[:, 1:])
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)
    
    df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
    ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
    df_tmp = pd.DataFrame(data = x_va, columns=col_names)
    ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
    df_tmp = pd.DataFrame(data = x_te, columns=col_names)
    ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')

    # remove random points from test dataset
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
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               np.asarray(x_na[:, 1:], dtype=np.float64), s_x
    
    
def load_synthetic_e1(ts=False, train_ratio=0.5, seed=0):
    df = timeseries_data.load_synthetic_data()
    
    df.index.name = 'timestamp'
    
    df.reset_index(inplace=True)
    col_names = df.columns
 
    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    x_al[:, 1:] = np.asarray(x_al[:, 1:], dtype=np.float64)
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed, shuffle=False)

    s_x = StandardScaler().fit(x_tr[:, 1:])
    x_tr[:, 1:] = s_x.transform(x_tr[:, 1:])
    x_va[:, 1:] = s_x.transform(x_va[:, 1:])
    x_te[:, 1:] = s_x.transform(x_te[:, 1:])

    if ts:
        
        df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
        ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
        df_tmp = pd.DataFrame(data = x_va, columns=col_names)
        ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
        df_tmp = pd.DataFrame(data = x_te, columns=col_names)
        ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')

        return ts_tr, ts_va, ts_te, 0, s_x
    
    else:
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               0, s_x
    

    
def load_synthetic_e2(ts=False, train_ratio=0.5, delete_ratio=0.5, seed=0):
    df = timeseries_data.load_synthetic_data()
    df.index.name = 'timestamp'
    
    df.reset_index(inplace=True)
    col_names = df.columns
 
    df_dim = df.shape[1]
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    
    x_al = np.asarray(df).copy().reshape(-1, df_dim)
    
    s_x = StandardScaler().fit(x_al[:, 1:])
    x_al[:, 1:] = s_x.transform(x_al[:, 1:])
    
    x_tr, x_te, = train_test_split(x_al, test_size=0.1, random_state=seed, shuffle=False)
    x_tr, x_va = train_test_split(x_tr, test_size=0.01, random_state=seed,  shuffle=False)
    
    df_tmp = pd.DataFrame(data = x_tr, columns=col_names)
    ts_tr = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
    df_tmp = pd.DataFrame(data = x_va, columns=col_names)
    ts_va = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')
        
    df_tmp = pd.DataFrame(data = x_te, columns=col_names)
    ts_te = TimeSeries.from_dataframe(df=df_tmp.reset_index(), time_col='timestamp')

    # remove random points from test dataset
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
        
        return np.asarray(x_tr[:, 1:], dtype=np.float64),\
               np.asarray(x_va[:, 1:], dtype=np.float64),\
               np.asarray(x_te[:, 1:], dtype=np.float64), \
               np.asarray(x_na[:, 1:], dtype=np.float64), s_x
    