import numpy
import h5py
import pandas

################################ Synthetic data #########################################################################################

def _proj(U, v):
    return numpy.dot(numpy.dot(v, U) / numpy.linalg.norm(U, axis=0), U.T)

def _gs(N):
    V = numpy.zeros((N, N))
    A = numpy.random.randn(N,N)
    #A = numpy.eye(N)
    for d in range(N):
        v = A[:,d]
        V[:,d] = v - _proj(V[:,:d], v)
        V[:,d] /= numpy.sqrt(numpy.sum(V[:,d] ** 2))
    return V


def _generate_heteroscedastic_data(T, Dx, Dz, Du, sigma_z = .02, sigma_x = .02):
    params_dict = {'T': T, 'Dx': Dx, 'Dz': Dz, 'Du': Du, 'sigma_z': sigma_z, 'sigma_x': sigma_x}
    C = numpy.random.randn(Dx, Dz)
    C /= numpy.sum(C, axis=0)[None] * .5
    U = _gs(Dx)[:Du].T
    w = 2 * numpy.random.randn(Du, Dz)
    #w /=  numpy.sum(numpy.abs(w), axis=1)[:,None]
    b_w = numpy.random.randn(Du)
    beta = 1e-2 * numpy.random.rand(Du)
    params_dict = {**params_dict, 'C': C, 'U': U, 'w': w, 'b_w': b_w, 'beta': beta}
    
    # Sample latent space
    z = numpy.zeros([Dz, T])
    noise_z = sigma_z * numpy.random.randn(Dz, T)
    # A = .99 * numpy.eye(Dz)
    # A[1,0] = .05
    # A[0,1] = -.05
    # b = numpy.zeros(Dz)
    # for t in range(1,T):
    #     z[:,t] = numpy.dot(A, z[:,t-1]) + b + noise_z[:,t-1]
    freq = 1 / (1000 * numpy.random.rand(Dz) + 500)
    phase = 2 * numpy.pi * numpy.random.rand(Dz)
    for idz in range(Dz): 
        z[idz] = 1*numpy.cos(2 * numpy.pi * numpy.arange(T) * freq[idz] + phase[idz]) + noise_z[idz]
    D_Sigma = 2 * beta[:,None] * (numpy.cosh(numpy.dot(w, z) + b_w[:,None]))
    x = numpy.zeros((Dx,T))
    mu_x = numpy.dot(C, z)
    noise_x = numpy.random.randn(Dx, T)
    for t in range(T):
        Sigma_x = sigma_x ** 2 * numpy.eye(Dx) + numpy.dot(numpy.dot(U, numpy.diag(D_Sigma[:,t])), U.T)
        L_x = numpy.linalg.cholesky(Sigma_x)
        x[:,t] = mu_x[:,t] + numpy.dot(L_x, noise_x[:,t])
    return x.T, z.T, params_dict

def load_synthetic_data(Dz:int = 2, Dx:int = 7, Du:int = 3, T:int = 4000, sigma_x: float=.01):
    var_names = ['x_%d' %i for i in range(Dx)]
    return pandas.DataFrame(data=_generate_heteroscedastic_data(T, Dx, Dz, Du, sigma_x=sigma_x)[0], columns=var_names)

################################ Airfoil data ###########################################################################################

def _load_pressure_ts(experiment_name, data_path = '../data/', piv=False, field_names=None, train_ratio=1):
    f = h5py.File(data_path + experiment_name + '/%s_pressure.mat' %experiment_name, 'r')
    if field_names is None:
        field_names = ['t', 'alpha_theory', 'alpha', 'd2alphadt2', 'dalphadt', 'phase', 'beta', 'phase_beta', 'Cl', 'Cm', 'Cp', 'stgx']  
    if piv:
        field = 'piv'
    else:
        field = 'res'
    pressure_dict = {}
    for name in field_names:
        if name == 'Cp':
            num_sensors = f[field][name].shape[1]
            for isensor in range(num_sensors):
                pressure_dict['Cp%d' % isensor] = f[field][name][:, isensor]
        elif name == 'stgx':
            stgx_df = pandas.read_csv(data_path + '%s/%s_stgdata.csv' %(experiment_name, experiment_name[5:]))
            pressure_dict[name] = stgx_df['stgx'].to_numpy()
        else:
            pressure_dict[name] = numpy.squeeze(f[field][name])
    f.close()
    if train_ratio==1:
        return pandas.DataFrame(pressure_dict)
    else:
        df = pandas.DataFrame(pressure_dict)
        exp_len = df.shape[0]
        train_len = int(train_ratio * exp_len)
        train_df = df[df.keys()][:train_len]
        test_df = df[df.keys()][train_len:]
        return train_df, test_df

def _load_pressure_sensor_pos(experiment_name, data_path = '../data/'):
    f = h5py.File(data_path + experiment_name + '/%s_pressure.mat' % experiment_name, 'r')
    pos_dict = {'xcp': numpy.squeeze(f['res']['xcp']), 'ycp': numpy.squeeze(f['res']['ycp'])}
    f.close()
    return pandas.DataFrame(pos_dict)

def _load_exp_params(experiment_name, data_path = '../data/'):
    f = h5py.File(data_path + experiment_name + '/%s_pressure.mat' %experiment_name, 'r')
    U0, chord_length, rho = f['param']['U0'][0][0], f['param']['c'][0][0], f['param']['rho'][0][0]
    f.close()
    return U0, chord_length, rho

def load_airfoil_data():
    pressure_df = _load_pressure_ts('ms033mpt006', data_path='../../data/')
    sensor_names = ['Cp%d' %i for i in range(0,36,4)]
    target_names = ['Cl', 'Cm']
    var_names = sensor_names + target_names
    return pressure_df[var_names][::8].to_numpy(), var_names

################################### Sunspot data ###############################################

def load_sunspot_data():
    df = pandas.read_csv('../../data/sunspots/monthly-sunspots.csv', parse_dates=True)
    var_names = ['monthly_sunspots']
    return numpy.array([df['Sunspots'].to_numpy()]).T, var_names

################################### OPSD data ###############################################

def load_opsd_data():
    de_energy_df = pandas.read_csv('../../data/opsd/time_series_60min_singleindex_filtered.csv')
    de_energy_df['utc_timestamp'] = pandas.to_datetime(de_energy_df['utc_timestamp'].astype(str).apply(lambda x: x[:10]))
    de_energy_df = de_energy_df.groupby('utc_timestamp').mean()
    de_weather_df = pandas.read_csv('../../data/opsd/weather_data_filtered.csv')
    de_weather_df['utc_timestamp'] = pandas.to_datetime(de_weather_df['utc_timestamp'].astype(str).apply(lambda x: x[:10]))
    de_weather_df = de_weather_df.groupby('utc_timestamp').mean()
    full_df = pandas.concat([de_energy_df[['DE_solar_generation_actual']], de_weather_df[['DE_temperature', 'DE_radiation_direct_horizontal', 'DE_radiation_diffuse_horizontal']]], axis=1)
    full_df = full_df.fillna(method='backfill')
    var_names = ['DE_solar_generation_actual'] + ['DE_temperature', 'DE_radiation_direct_horizontal', 'DE_radiation_diffuse_horizontal']
    return full_df[1:-1].to_numpy(), var_names