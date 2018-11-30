"""
:Author: yannick schwartz, dohmatob elvis dopgima
:Synopsis: Minimal script for preprocessing single-subject data
+ GLM with nistats
"""

# standard imports
import sys
import os
import time

from nipype.interfaces.fsl import ExtractROI

from pypreprocess.nipype_preproc_fsl_utils import do_subject_preproc
from pypreprocess.subject_data import SubjectData

dataset_dir = '/neurospin/psy_sbox/hbn/RU/sourcedata/'

# preprocess the data
subject_id = "sub-NDARDV222JZ6"
anat = os.path.join(dataset_dir, subject_id, 'anat',
                    'sub-NDARDV222JZ6_acq-HCP_T1w.nii.gz')
func = os.path.join(dataset_dir, subject_id, 'func',
                    'sub-NDARDV222JZ6_task-rest_run-1_bold.nii.gz')
subject_data = SubjectData(
    output_dir=os.path.join("/volatile/home/kamalakar/work/hbn",
                            "pypreprocess_output", subject_id),
    subject_id=subject_id, func=func,
    anat=anat)
subject_data = do_subject_preproc(subject_data, do_bet=True, do_mc=True,
                                  do_coreg=True, do_normalize=True,
                                  do_ica_aroma=True,
                                  do_smooth=True, fwhm=5.)
