from nilearn import datasets
import ABIDEParser as Reader
import os
import shutil

pipeline = 'cpac'

num_subjects = 871  
root_folder = './'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')

files = ['rois_ho']

filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_ho': 'rois_ho.1D'}

if not os.path.exists(data_folder): os.makedirs(data_folder)
shutil.copyfile('./data/subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))



subject_IDs = Reader.get_ids(num_subjects)
subject_IDs = subject_IDs.tolist()


time_series = Reader.get_timeseries(subject_IDs, 'ho')

for i in range(len(subject_IDs)):
        Reader.subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')

