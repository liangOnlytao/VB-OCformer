import os
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
import onnxruntime as ort

from torch.utils.data import DataLoader

from src.data import Dataset
from src.utils import set_seed

from scipy import stats
from cftime import date2num
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_nc_data(nc_array):
    nc_data = nc_array.data
    nc_data[np.where(nc_array.mask)] = np.nan
    return nc_data

def main(args, random_state):
    satellite = args.satellite
    device_id = args.device_id

    rrs_data_path = f'./inputs_rrs/{satellite}'
    rrs_nc_files = sorted(glob.glob(f'{rrs_data_path}/*.nc', recursive=True))
    date = os.path.basename(rrs_nc_files[0]).split('.')[1]

    target_data_path = f'./output_chlor_a/{satellite}'
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    target_nc_file = f'{target_data_path}/{satellite.lower()}_chlor_a_{date}.nc'
    if not os.path.exists(target_nc_file):
        set_seed(random_state)

        dataset_path = f'./dataset'
        dataset_file = os.path.join(dataset_path, f'{satellite.lower()}.npz')
        np_data = np.load(dataset_file)
        dataset = {}
        for key in np_data:
            dataset[key] = np_data[key]
        dataset = pd.DataFrame.from_dict(dataset)
        features_train, _, target_train, _ = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, [-1]], test_size=0.2, train_size=0.8,
                                                              random_state=random_state, shuffle=True, stratify=None)

        features_scaler = StandardScaler().fit(features_train)
        target_scaler = StandardScaler().fit(target_train)

        onnx_model_path = f'./trained models/{satellite.lower()}.onnx'
        sess_options = ort.SessionOptions()
        if device_id is not None:
            providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(
                path_or_bytes = onnx_model_path,
                sess_options = sess_options,
                providers = providers)
        
        def inference(ort_session, X, device_id=None):
            io_binding = ort_session.io_binding()
            if device_id is not None:
                X_ortvalue = ort.OrtValue.ortvalue_from_numpy(X.detach().numpy(), 'cuda', device_id)
                io_binding.bind_ortvalue_input(ort_session.get_inputs()[0].name, X_ortvalue)
            else:
                io_binding.bind_cpu_input(ort_session.get_inputs()[0].name, X.detach().numpy())
            io_binding.bind_output(ort_session.get_outputs()[0].name)
            ort_session.run_with_iobinding(io_binding)
            y = io_binding.copy_outputs_to_cpu()[0]

            return y
        
        ci = 0.95
        num_samples = 10

        mask_nc_file = f"./mask_{9 if satellite=='SeaWiFS' else 4}km.nc"
        mask_nc_data = nc.Dataset(mask_nc_file)
        longitude_data = mask_nc_data['longitude'][:].data
        latitude_data = mask_nc_data['latitude'][:].data
        mask_data = mask_nc_data['mask'][:].data

        bands = sorted([column for column in dataset.columns if 'Rrs' in column])

        features = {band: None for band in bands}
        for band in bands:
            for rrs_nc_file in rrs_nc_files:
                if band in os.path.basename(rrs_nc_file):
                    break
            rrs_nc_data = nc.Dataset(rrs_nc_file)
            features[band] = get_nc_data(rrs_nc_data[band][:])
        for band in bands:
            index = np.where((~np.isnan(features[band])) & (features[band] <= 0))
            for band in bands:
                features[band][index] = np.nan
        
        index = np.where(~np.isnan(features[bands[0]]))
        features_use = {band: features[band][index] for band in bands}
        features_use = pd.DataFrame.from_dict(features_use)
        features_scaled = features_scaler.transform(features_use)

        predict_dataset = Dataset(Xs=features_scaled, ys=np.zeros((features_scaled.shape[0])))
        predict_dataloader = DataLoader(dataset = predict_dataset,
                                        batch_size = 25600,
                                        shuffle = False,
                                        num_workers = 0,
                                        pin_memory = True,
                                        drop_last = False)
        
        with torch.no_grad():
            for idx, (X, _) in enumerate(predict_dataloader):
                target_predicted = np.array([inference(ort_session, X, device_id) for _ in range(num_samples)])
                if idx == 0:
                    target_predicted_all = target_predicted
                else:
                    target_predicted_all = np.hstack((target_predicted_all, target_predicted))
        
        target_predicted_all = target_scaler.inverse_transform(target_predicted_all)
        target_predicted_all = 10 ** target_predicted_all.T

        target_predicted_mean = np.mean(target_predicted_all, axis=1)

        t_value = stats.t.ppf(q=((1 + ci) / 2), df=num_samples - 1)
        target_predicted_std = target_predicted_all.std(axis=1) * t_value
        target_predicted_cv = target_predicted_std / target_predicted_mean * 100

        chlor_a_data = np.zeros((latitude_data.shape[0], longitude_data.shape[0]))
        chlor_a_data[:] = np.nan
        chlor_a_data[index] = target_predicted_mean
        chlor_a_data[mask_data==0] = np.nan
        chlor_a_data = np.ma.masked_invalid(chlor_a_data)

        chlor_a_std_data = np.zeros((latitude_data.shape[0], longitude_data.shape[0]))
        chlor_a_std_data[:] = np.nan
        chlor_a_std_data[index] = target_predicted_std
        chlor_a_std_data[mask_data==0] = np.nan
        chlor_a_std_data = np.ma.masked_invalid(chlor_a_std_data)
            
        chlor_a_cv_data = np.zeros((latitude_data.shape[0], longitude_data.shape[0]))
        chlor_a_cv_data[:] = np.nan
        chlor_a_cv_data[index] = target_predicted_cv
        chlor_a_cv_data[mask_data==0] = np.nan
        chlor_a_cv_data = np.ma.masked_invalid(chlor_a_cv_data)


        nc_file_write = nc.Dataset(target_nc_file, 'w', format='NETCDF4')

        nc_file_write.createDimension('longitude', len(longitude_data))
        nc_file_write.createDimension('latitude', len(latitude_data))
        nc_file_write.createDimension('time', 1)

        longitude = nc_file_write.createVariable('longitude', 'f4', ('longitude',), zlib=True, complevel=4)
        longitude.long_name = f'longitude'
        longitude.units = f'degrees_east'

        latitude = nc_file_write.createVariable('latitude', 'f4', ('latitude',), zlib=True, complevel=4)
        latitude.long_name = f'latitude'
        latitude.units = f'degrees_north'

        time = nc_file_write.createVariable('time', 'i4', ('time',), zlib=True, complevel=4)
        time.long_name = f'time'
        time.units = f'days since 1950-01-01T00:00:00Z'

        chlor_a = nc_file_write.createVariable('chlor_a', 'f8', ('latitude', 'longitude'), zlib=True, complevel=4, fill_value=-32767)
        chlor_a.long_name = f'Chlorophyll-a concentration'
        chlor_a.units = f'mg m^-3'

        chlor_a_std = nc_file_write.createVariable('chlor_a_std', 'f8', ('latitude', 'longitude'), zlib=True, complevel=4, fill_value=-32767)
        chlor_a_std.long_name = f'The standard deviation of chlorophyll-a concentration'
        chlor_a_std.units = f'mg m^-3'

        chlor_a_cv = nc_file_write.createVariable('chlor_a_cv', 'f8', ('latitude', 'longitude'), zlib=True, complevel=4, fill_value=-32767)
        chlor_a_cv.long_name = f'The coefficient of variation of chlorophyll-a concentration'
        chlor_a_cv.units = f'%'

        longitude[:] = longitude_data
        latitude[:] = latitude_data
        time[:] = date2num(dates=datetime(int(date[:4]), int(date[4:6]), int(date[6:])), units='days since 1950-01-01T00:00:00Z')
        chlor_a[:] = chlor_a_data
        chlor_a_std[:] = chlor_a_std_data
        chlor_a_cv[:] = chlor_a_cv_data

        nc_file_write.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--satellite', type=str, choices=['MERIS', 'MODIS-Aqua', 'MODIS-Terra', 'SeaWiFS', 'SNPP-VIIRS'], required=True,
                        help='Select one of the following sensors: MERIS, MODIS-Aqua, MODIS-Terra, SeaWiFS, SNPP-VIIRS.')
    parser.add_argument('--device_id', type=int, required=False)
    args = parser.parse_args()

    global_seed = 0
    rstate = np.random.default_rng(seed=global_seed)
    random_state = rstate.integers(2**31 - 1)
    main(args, random_state)