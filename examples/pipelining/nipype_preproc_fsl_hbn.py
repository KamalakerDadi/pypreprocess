"""
:Author: yannick schwartz, dohmatob elvis dopgima
:Synopsis: Minimal script for preprocessing single-subject data
+ GLM with nistats
"""

# standard imports
import sys
import os
import glob
import time
from itertools import izip
from os.path import join
import pandas as pd

from nipype.interfaces.fsl import ExtractROI

from pypreprocess.nipype_preproc_fsl_utils import do_subject_preproc
from pypreprocess.subject_data import SubjectData


def _do_subject_preproc(anat, funcs, subject_id, output_dir):

    for func in funcs:
        # Grab folder name to save for each run and each task
        unique_id = os.path.splitext(
            os.path.splitext(os.path.split(func)[1])[0])[0]
        subject_data = SubjectData(
            output_dir=join(output_dir, 'pypreprocess_output',
                            subject_id, unique_id),
            subject_id=subject_id, func=func, anat=anat)
        subject_data = do_subject_preproc(subject_data, do_bet=True,
                                          do_mc=True, do_coreg=True,
                                          do_normalize=True, do_ica_aroma=True,
                                          do_smooth=True, fwhm=5.)
    return

data_dir = '/neurospin/psy_sbox/hbn/RU/sourcedata/'
subjects_paths = glob.glob(join(data_dir, 'sub-*'))

output_base_dir = '/media/kr245263/4C9A6E0E9A6DF53C/hbn'

anat_default_path = '{}_acq-HCP_T1w.nii.gz'

for subject_path in subjects_paths:
    subject_id = os.path.split(subject_path)[1]
    this_anat = join(subject_path, 'anat',
                     anat_default_path.format(subject_id))
    if os.path.exists(this_anat):
        funcs = glob.glob(join(subject_path, 'func',
                               '*_bold.nii.gz'))
        if len(funcs) > 0:
            _do_subject_preproc(anat=this_anat, funcs=funcs,
                                subject_id=subject_id,
                                output_dir=output_base_dir)
