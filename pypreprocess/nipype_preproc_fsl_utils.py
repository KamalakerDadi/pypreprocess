"""
Author: Bertrand Thirion, Alexandre Abraham, DOHMATOB Elvis Dopgima, Kamalaker Dadi

"""

import os
import warnings
import subprocess
import logging
import nipype.interfaces.fsl as fsl
import numpy as np
from nipype.caching import Memory as NipypeMemory
from nipype.interfaces.fsl import utils
from sklearn.externals.joblib import Memory as JoblibMemory
from nilearn._utils.compat import _basestring
from .reporting.preproc_reporter import generate_preproc_undergone_docstring
from .subject_data import SubjectData

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
FSL_T1_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
FSL_T1_MASK = "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
FSL_WARP_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_2mm.nii.gz"

_logger = logging.getLogger("pypreprocess")
_INTERFACE_ERROR_MSG = ("Interface {0} (version {1}) failed with "
                        "parameters\n{2} \nand with error\n{3}.")


def _update_interface_inputs(**kwargs):
    """Update kwargs interface inputs stored in 'interface_kwargs'.
    """
    interface_kwargs = kwargs.pop("interface_kwargs", {})
    _kwargs = kwargs
    _kwargs.update(interface_kwargs)
    return _kwargs


def _get_file_ext(filename):
    parts = filename.split('.')

    return parts[0], ".".join(parts[1:])


def _get_output_filename(input_filename, output_dir, output_prefix='',
                         ext=None):
    if isinstance(input_filename, _basestring):
        if ext is not None:
            ext = "." + ext if not ext.startswith('.') else ext
            input_filename = _get_file_ext(input_filename)[0] + ext

        return os.path.join(output_dir,
                            output_prefix + os.path.basename(input_filename))
    else:
        return [_get_output_filename(x, output_dir,
                                     output_prefix=output_prefix)
                for x in input_filename]


def do_fsl_merge(in_files, output_dir, output_prefix='merged_',
                 cmd_prefix="fsl5.0-"
                 ):
    output_filename = _get_output_filename(in_files[0], output_dir,
                                           output_prefix=output_prefix,
                                           ext='.nii.gz')

    cmdline = "%sfslmerge -t %s %s" % (cmd_prefix, output_filename,
                                       " ".join(in_files))
    print(cmdline)
    print(subprocess.check_output(cmdline))

    return output_filename


def _cache_interface(subject_data, function):
    """Caching Nipype interface

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like scratch - which is a directory to cache the intermediary
        results)

    function : callable
        The callable function to cache. Eg fsl.ExtractROI, fsl.MCFLIRT

    Returns
    -------
    function : callable
        An input function is cached function and returned
    """
    cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    subject_data.mem = NipypeMemory(base_dir=cache_dir)
    function = subject_data.mem.cache(function)
    return subject_data, function


def _error_if_node_is_failed(interface):
    """Logging error

    Parameters
    ----------
    interface : An interface input having an arguments in
        interface, version, inputs, runtime.traceback
    """
    _logger.error(_INTERFACE_ERROR_MSG.format(
        interface.interface, interface.version,
        interface.inputs, interface.runtime.traceback))
    return


def _do_subject_extract_roi(subject_data, caching, cmd_prefix,
                            t_min, hardlink_output, **kwargs):
    """Removal of number of dummy scans from functional MRI data.

    Helper function to run fsl.ExtractROI

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    t_min : int
        Number of volumes to remove.

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    Returns
    -------
    subject_data : `SubjectData` instance

        - func: A functional image where dummy scans are removed. This is
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_roi.nii.gz

    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin dummy scans "
                      "removal step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, extract = _cache_interface(subject_data,
                                                 fsl.ExtractROI)
    else:
        extract = fsl.ExtractROI().run

    if not fsl.ExtractROI._cmd.startswith("fsl"):
        fsl.ExtractROI._cmd = cmd_prefix + fsl.ExtractROI._cmd

    extract_roi_results = extract(**_update_interface_inputs(
        in_file=(subject_data.func[0]
                 if isinstance(subject_data.func, list)
                 else subject_data.func),
        t_min=t_min,
        t_size=-1,
        interface_kwargs=kwargs))

    # failed node
    if extract_roi_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(extract_roi_results)
        return subject_data

    # collect output
    subject_data.func = extract_roi_results.outputs.roi_file

    subject_data.nipype_results['dummy scans'] = extract_roi_results

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_bet(subject_data, caching, cmd_prefix,
                    hardlink_output, **kwargs):
    """FSL BET extraction on anatomical/structural T1 data.

    Helper function to run fsl.BET

    By default, mask is set as True to output the mask.

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like anat - which is an anatomical MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    Returns
    -------
    subject_data : `SubjectData` instance

        - anat : A BET extracted anatomical image. This is
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_brain.nii.gz

        - mask : A BET extracted mask image. Saved as xxxx_brain_mask.nii.gz

    """
    if not subject_data.anat:
        warnings.warn("subject_data.anat=%s (empty); skippin Brain Extraction"
                      % (subject_data.anat), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, bet = _cache_interface(subject_data, fsl.BET)
    else:
        bet = fsl.BET().run

    if not fsl.BET._cmd.startswith("fsl"):
        fsl.BET._cmd = cmd_prefix + fsl.BET._cmd

    bet_results = bet(**_update_interface_inputs(
        in_file=subject_data.anat, mask=True, interface_kwargs=kwargs))

    # failed node
    if bet_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(bet_results)
        return subject_data

    # collect output
    subject_data.anat = bet_results.outputs.out_file
    subject_data.anat_mask = bet_results.outputs.mask_file

    subject_data.nipype_results['bet'] = bet_results

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_mcflirt(subject_data, caching, register_to_mean, cost,
                        cmd_prefix, hardlink_output, report, **kwargs):
    """Runs motion correction using FSL MCFLIRT on functional MRI data.

    Helper function to run fsl.MCFLIRT to register to mean volume (by default).

    By default, the cost function it uses is 'mutualinfo'

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    register_to_mean : bool
        Register to mean volume.

    cost : str ('mutualinfo', 'woods', 'corratio', 'normcorr', 'normmi',
                'leastsquares')
        Cost function to optimize.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    report : bool
        If True, rotation and translation curves are reported to html report.

    Returns
    -------
    subject_data : `SubjectData` instance

        - func : A motion corrected functional image. This is
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_roi_mcf.nii.gz

        - motion correction parameter : A .par file contains translation and
        rotation each in (x, y, z)

    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin mcflirt "
                      "realignment step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, mcflirt = _cache_interface(subject_data,
                                                 fsl.MCFLIRT)
    else:
        mcflirt = fsl.MCFLIRT().run

    if not fsl.MCFLIRT._cmd.startswith("fsl"):
        fsl.MCFLIRT._cmd = cmd_prefix + fsl.MCFLIRT._cmd

    mcflirt_results = mcflirt(**_update_interface_inputs(
        in_file=subject_data.func[0], mean_vol=register_to_mean,
        cost=cost, save_mats=True, save_plots=True,
        interface_kwargs=kwargs))

    # failed node
    if mcflirt_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(mcflirt_results)
        return subject_data

    # collect output
    subject_data.func = mcflirt_results.outputs.out_file
    subject_data.realignment_parameters = \
        mcflirt_results.outputs.par_file
    if isinstance(subject_data.realignment_parameters, _basestring):
        assert subject_data.n_sessions == 1
        subject_data.realignment_parameters = [
            subject_data.realignment_parameters]

    subject_data.nipype_results['realign'] = mcflirt_results
    if isinstance(subject_data.func, _basestring):
        assert subject_data.n_sessions == 1
        subject_data.func = [subject_data.func]

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_realignment_thumbnails()

    return subject_data.sanitize()


def _do_subject_fsl_motion_outliers(subject_data, caching,
                                    fsl_motion_outliers_metric,
                                    cmd_prefix, hardlink_output, **kwargs):
    """Finds motion outliers on functional MRI data.

    Helper function to run fsl.MotionOutliers.

    By default, the Frame Displacement FD with Root Mean Square matrix
    is estimated ('fdrms').

    There is no html reporting for this run but a simple png file is saved
    to subject_data.output_dir

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    fsl_motion_outliers_metrics : str ('refrms', 'dvars', 'refmse', 'fd',
                                       'fdrms')
        metrics to choose to compute motion outliers.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    report : bool
        If True, rotation and translation curves are reported to html report.

    Returns
    -------
    subject_data : `SubjectData` instance

        - out_file : A motion outliers values in .txt is generated and
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_outliers.txt

    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin compute motion "
                      "outliers step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, motion_outliers = _cache_interface(subject_data,
                                                         utils.MotionOutliers)
    else:
        motion_outliers = utils.MotionOutliers().run

    if not fsl.MCFLIRT._cmd.startswith("fsl"):
        fsl.MCFLIRT._cmd = cmd_prefix + fsl.MCFLIRT._cmd

    motion_outliers_results = motion_outliers(**_update_interface_inputs(
        in_file=subject_data.func[0], metric=fsl_motion_outliers_metric,
        interface_kwargs=kwargs))

    # failed node
    if motion_outliers_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(motion_outliers_results)
        return subject_data

    # collect output
    subject_data.motion_outliers = motion_outliers_results.outputs.out_metric_values
    subject_data.mo_plot = motion_outliers_results.outputs.out_metric_plot
    if isinstance(subject_data.motion_outliers, _basestring):
        assert subject_data.n_sessions == 1
        subject_data.motion_outliers = [subject_data.motion_outliers]

    subject_data.nipype_results['motion_outliers'] = motion_outliers_results

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_coregister(subject_data, caching, cmd_prefix,
                           hardlink_output, report, **kwargs):
    """FLIRT based co-registration of functional data to anatomical data.

    Helper function to run fsl.FLIRT to register func2struct.

    By default, 6 dof is used for this co-registration

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    report : bool
        If True, the alignment of anatomical image is overlayed on mean
        functional image for visual inspection alongside with other runs
        in html report.

    Returns
    -------
    subject_data : `SubjectData` instance

        - out_file : A motion outliers values in .txt is generated and
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_outliers.txt

    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin flirt "
                      "coregistration step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, flirt = _cache_interface(subject_data, fsl.FLIRT)
    else:
        flirt = fsl.FLIRT().run

    if not fsl.FLIRT._cmd.startswith("fsl"):
        fsl.FLIRT._cmd = cmd_prefix + fsl.FLIRT._cmd

    coreg_results = flirt(**_update_interface_inputs(
        in_file=subject_data.func[0], reference=subject_data.anat,
        dof=6, interface_kwargs=kwargs))

    # failed node
    if coreg_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(coreg_results)
        return subject_data

    subject_data.nipype_results['coreg'] = coreg_results
    # collect output
    subject_data.func = coreg_results.outputs.out_file
    subject_data.func2structmat = coreg_results.outputs.out_matrix_file

    if isinstance(subject_data.func, _basestring):
        subject_data.func = [subject_data.func]
    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_coregistration_thumbnails()
    return subject_data.sanitize()


def _do_subject_normalize(subject_data, caching, cmd_prefix,
                          hardlink_output, report, do_coreg,
                          **kwargs):
    """FNIRT based non-linear registration of functional data
       to FSL T1 template (via structural scan).

    This helper function does the following steps:

        1. Runs fsl.FLIRT to register structural scan to FSL T1 template.
           Saves affine transformation matrix file. Outputs registered
           structural scan.

        2. Then, runs non-linear transformation on structural scan to save
           non-linear warp file.

        3. At last runs applywarp using pre-computed matrix file from step 1
           and warp from step 2, on functional data using FSL T1 template
           as the reference. Outputs functional data.

    Please note that this step is dependent on prior co-registration of
    structural scan to functional MRI image. If do_coreg is given as False,
    then this step is skipped.

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like anat, func - which is a structural scan and functional MRI image)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    report : bool
        If True, the alignment of anatomical image is overlayed on mean
        functional image for visual inspection alongside with other runs
        in html report.

    do_coreg : bool
        This step indicates whether registration of structural scan to
        functional MRI image is chosen or not. If not chosen, then
        this normalization step is skipped.

    Returns
    -------
    subject_data : `SubjectData` instance

        - out_file : A structural scan and functional MRI image
        registered to FSL T1 template. This image is
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_warp.nii.gz

    """
    if not subject_data.func[0] or not do_coreg:
        warnings.warn("subject_data.func=%s (empty); skippin fnirt and "
                      "applywarp based normalization step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, flirt = _cache_interface(subject_data, fsl.FLIRT)
    else:
        flirt = fsl.FLIRT().run

    if not fsl.FLIRT._cmd.startswith("fsl"):
        fsl.FLIRT._cmd = cmd_prefix + fsl.FLIRT._cmd

    # Step 1:

    norm_results = flirt(**_update_interface_inputs(
        in_file=subject_data.anat, reference=FSL_T1_TEMPLATE,
        interface_kwargs=kwargs))

    # failed node
    if norm_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(norm_results)
        return subject_data

    subject_data.nipype_results['flirt2'] = norm_results
    # collect output
    subject_data.anat = norm_results.outputs.out_file

    if caching:
        # prepare for smart-caching
        subject_data, fnirt = _cache_interface(subject_data, fsl.FNIRT)
    else:
        fnirt = fsl.FNIRT().run

    if not fsl.FNIRT._cmd.startswith("fsl"):
        fsl.FNIRT._cmd = cmd_prefix + fsl.FNIRT._cmd

    # Step 2:

    struct_file = subject_data.nipype_results['bet'].inputs['in_file']
    fnirt_results = fnirt(**_update_interface_inputs(
        affine_file=norm_results.outputs.out_matrix_file,
        ref_file=FSL_T1_TEMPLATE,
        in_file=struct_file, config_file='T1_2_MNI152_2mm',
        field_file=True, fieldcoeff_file=True,
        interface_kwargs=kwargs))
    subject_data.nipype_results['fnirt'] = fnirt_results

    subject_data.fnirt_warp_file = fnirt_results.outputs.field_file

    if caching:
        # prepare for smart-caching
        subject_data, applywarp = _cache_interface(subject_data,
                                                   fsl.ApplyWarp)
    else:
        applywarp = fsl.ApplyWarp().run

    if not fsl.ApplyWarp._cmd.startswith("fsl"):
        fsl.ApplyWarp._cmd = cmd_prefix + fsl.ApplyWarp._cmd

    # Step 3:

    # Apply final transform
    func2structmat = subject_data.nipype_results['coreg'].outputs.out_matrix_file
    applywarp_results = applywarp(**_update_interface_inputs(
        ref_file=FSL_WARP_TEMPLATE, mask_file=FSL_T1_MASK,
        in_file=subject_data.nipype_results['coreg'].inputs["in_file"],
        field_file=fnirt_results.outputs.field_file,
        premat=func2structmat, interface_kwargs=kwargs))
    subject_data.nipype_results['applywarp'] = applywarp_results
    subject_data.func = applywarp_results.outputs.out_file

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_normalization_thumbnails()
    return subject_data.sanitize()


def _do_subject_smoothing(subject_data, fwhm, caching, cmd_prefix,
                          hardlink_output, report, **kwargs):
    """FSL Smoothing functional MRI image.

    Helper function to run fsl.Smooth to smooth functional image
    of specified fwhm.

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    fwhm : float
        gaussian kernel fwhm, will be converted to sigma in mm (not voxels)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    report : bool
        If True, the alignment of anatomical image is overlayed on mean
        functional image for visual inspection alongside with other runs
        in html report.

    Returns
    -------
    subject_data : `SubjectData` instance

        - out_file : A smoothed functional image is generated and
        saved in subject_data.output_dir if hardlink_output is True with name
        at the end prefixed as xxxx_smooth.nii.gz

    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin smoothing "
                      "step " % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, smooth = _cache_interface(subject_data, fsl.Smooth)
    else:
        smooth = fsl.Smooth().run

    if not fsl.Smooth._cmd.startswith("fsl"):
        fsl.Smooth._cmd = cmd_prefix + fsl.Smooth._cmd

    smooth_results = smooth(**_update_interface_inputs(
        in_file=subject_data.func[0], fwhm=fwhm,
        interface_kwargs=kwargs))

    # failed node
    if smooth_results.outputs is None:
        subject_data.failed = True
        _error_if_node_is_failed(smooth_results)
        return subject_data

    subject_data.nipype_results['smooth'] = smooth_results
    # collect output
    subject_data.func = smooth_results.outputs.smoothed_file

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_smooth_thumbnails()
    return subject_data.sanitize()


def _do_subject_ica_aroma(subject_data, ica_aroma_denoise_type,
                          caching, hardlink_output, **kwargs):
    """ Performs ICA-AROMA on smoothed functional image
        (i.e. 'ICA-based Automatic Removal Of Motion Artifacts').

        Please set do_coreg=True and do_normalize=True to run this
        step. ICA Aroma requires functional to structural scan output
        matrix file from do_coreg step and non-linear FNIRT warp file
        from do_normalize step.

        A data-driven method to identify and remove motion-related independent
        components from fMRI data.

        See link for further documentation:
            https://github.com/maartenmennes/ICA-AROMA

        Runs through nipype but ICA-AROMA package needs to be installed before
        and this package is neither installed through nipype nor pypreprocess.

        Steps to setup ICA-Aroma:

        1. Download the latest package from
           https://github.com/maartenmennes/ICA-AROMA/releases

        2. Extract the .zip or .tar.gz file in a directory

        3. source the path to directory in .bashrc
           export PATH=/path/to/ICA-AROMA-0.4.4-beta:$PATH

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like func - which is a functional MRI image)

    ica_aroma_denoise_type : ('nonaggr' or 'aggr' or 'both' or 'no',
        nipype default value: nonaggr)
        Type of denoising strategy:
            -no: only classification, no denoising
            -nonaggr (default): non-aggresssive denoising, i.e. partial
            component regression
            -aggr: aggressive denoising, i.e. full component regression
            -both: both aggressive and non-aggressive denoising (two outputs)

    caching : bool
        If caching needs to be done or not. If yes, results are stored
        using NipypeMemory.

    hardlink_output : bool
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    Returns
    -------
    subject_data : `SubjectData` instance

        - out_file : A ICA-Aroma denoised functional image is generated and
        saved in subject_data.output_dir if hardlink_output is True with name
        at the start prefixed as denoised_func_data_xxxx.nii.gz
        xxxx depends on the denoising strategy chosen.

    Notes
    -----
    ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from
    fMRI data.
    Pruim RHR, Mennes M, van Rooij D, Llera A, Buitelaar JK, Beckmann CF.
    Neuroimage 2015.
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skipping ICA-AROMA "
                      "step " % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        subject_data, ica_aroma = _cache_interface(subject_data, fsl.ICA_AROMA)
    else:
        ica_aroma = fsl.ICA_AROMA().run

    func2structmat = subject_data.nipype_results['coreg'].outputs.out_matrix_file
    field_file = subject_data.nipype_results['fnirt'].outputs.field_file
    ica_aroma_results = ica_aroma(**_update_interface_inputs(
        in_file=subject_data.func[0], mat_file=func2structmat,
        fnirt_warp_file=field_file, mask=FSL_T1_MASK,
        motion_parameters=subject_data.realignment_parameters[0],
        denoise_type=ica_aroma_denoise_type, interface_kwargs=kwargs))

    if ica_aroma_results is not None:
        if ica_aroma_results.outputs.nonaggr_denoised_file is not None:
            subject_data.nonaggr_denoised_file = ica_aroma_results.outputs.nonaggr_denoised_file
        elif ica_aroma_results.outputs.aggr_denoised_file is not None:
            subject_data.aggr_denoised_file = ica_aroma_results.outputs.aggr_denoised_file
        else:
            warnings.warn("ICA aroma denoised files are not found to assign to "
                          "subject_data.", stacklevel=2)
    else:
        warnings.warn("ICA aroma denoising failed. Return subjec_data as it is.",
                      stacklevel=2)
        return subject_data

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def do_subject_preproc(subject_data,
                       remove_dummy_scans=True,
                       n_dummy_scans=0,
                       do_bet=True,
                       do_mc=True,
                       register_to_mean=True,
                       cost='mutualinfo',
                       do_fsl_motion_outliers=True,
                       fsl_motion_outliers_metric='fdrms',
                       do_coreg=True,
                       do_normalize=True,
                       do_smooth=True,
                       fwhm=0.,
                       do_ica_aroma=False,
                       ica_aroma_denoise_type='nonaggr',
                       cmd_prefix="fsl5.0-",
                       report=True,
                       parent_results_gallery=None,
                       hardlink_output=True,
                       caching=True,
                       **kwargs
                       ):
    """
    Preprocesses subject data using FSL.

    Parameters
    ----------
    subject_data : `SubjectData` instance
        object that encapsulates the data for the subject (should have fields
        like anat, func, output_dir, subject_id.
        anat - a structural image T1
        func - functional MRI image
        output_dir - To save the outputs and cache. For our convenience,
                     if there are more runs, then append each run with this
                     output directory so each run is saved separately.
        subject_id - Subject id
        Refer to documentation of `SubjectData` class in pypreprocess/subject_data.py

    remove_dummy_scans : bool, default=True
        Skip dummy scans from functional MRI image. Switch to False if dummy
        scans step doesn't need to be done.

    n_dummy_scans : int, default=0
        Number of dummy scans to remove.
        Please read documentation of ExtractROI in FSL for full details.

    do_bet : bool, default=True
        Whether to do Brain extraction or not on structural scan.

    do_mc : bool, default=True
        Whether to do MCFLIRT motion correction or not on functional MRI image.

    register_to_mean : bool, default=True
        Whether register to mean volume used when `do_mc` is True.

    cost : str, default='mutualinfo'
        Cost function to optimize. Cost functions available are
        ('mutualinfo', 'woods', 'corratio', 'normcorr', 'normmi', 'leastsquares')
        Used when `do_mc` is True.

    do_fsl_motion_outliers : bool, default=True
        To detect timepoints in functional MRI image that have been induced
        by large motion. See parameter `fsl_motion_outliers_metric` for metrics
        available to choose to detect motion outliers.

    fsl_motion_outliers_metric : str default='fdrms'
        'refrms' - RMS intensity difference to reference volume as metric,
        'refmse' - Mean Square Error version of refrms (used in original
                   version of fsl_motion_outliers),
        'dvars' - DVARS as metric,
        'fd' - frame displacement,
        'fdrms' - FD with RMS matrix calculation
    Used when `do_fsl_motion_outliers` is True

    do_coreg : bool, default=True
        FLIRT based co-registration of functional MRI image to structural scan.

    do_normalize : bool, default=True
        FNIRT based non-linear registration of functional data
        to FSL T1 template (via structural scan).

        This helper function does the following steps:

        1. Runs fsl.FLIRT to register structural scan to FSL T1 template.
           Saves affine transformation matrix file. Outputs registered
           structural scan.

        2. Then, runs non-linear transformation on structural scan to save
           non-linear warp file.

        3. At last runs applywarp using pre-computed matrix file from step 1
           and warp from step 2, on functional data using FSL T1 template
           as the reference. Outputs functional data.

    NOTE: Do not put `do_coreg` to False if you want to use normalize.

    do_smooth : bool, default=True
        Whether to smooth functional MRI image.

    fwhm : float, default=0.
        gaussian kernel fwhm, will be converted to sigma in mm (not voxels)

    NOTE: A simple warning is raised if not modified from fwhm=0. to non-zero

    do_ica_aroma : bool, default=False
        This step takes fair amount of time. To do this step
        `do_mc`, `do_coreg`, `do_normalize` needs to be done before to take
        ouputs as inputs for ICA AROMA.

        Performs ICA-AROMA on smoothed functional image
        (i.e. 'ICA-based Automatic Removal Of Motion Artifacts').

        Please set do_coreg=True and do_normalize=True to run this
        step. ICA Aroma requires functional to structural scan output
        matrix file from do_coreg step and non-linear FNIRT warp file
        from do_normalize step.

        A data-driven method to identify and remove motion-related independent
        components from fMRI data.

        See link for further documentation:
            https://github.com/maartenmennes/ICA-AROMA

        Runs through nipype but ICA-AROMA package needs to be installed before
        and this package is neither installed through nipype nor pypreprocess.

        Steps to setup ICA-Aroma:

        1. Download the latest package from
           https://github.com/maartenmennes/ICA-AROMA/releases

        2. Extract the .zip or .tar.gz file in a directory

        3. source the path to directory in .bashrc
           export PATH=/path/to/ICA-AROMA-0.4.4-beta:$PATH

    NOTE: Look for `ica_aroma_denoise_type` for denoising strategies.

    ica_aroma_denoise_type : str, default='nonaggr'
        Denoising strategies to use with ICA AROMA. Valid strategies
        are ('nonaggr' or 'aggr' or 'both' or 'no')
        Type of denoising strategy:
            -no: only classification, no denoising
            -nonaggr (default): non-aggresssive denoising, i.e. partial
            component regression
            -aggr: aggressive denoising, i.e. full component regression
            -both: both aggressive and non-aggressive denoising (two outputs)

    cmd_prefix : str
        Command prefix for FSL command lines "fsl5.0-".

    report : bool, default=True
        If True, the alignment of anatomical image is overlayed on mean
        functional image for visual inspection alongside with other runs
        in html report.

    parent_results_gallery default=None
        reporting.base_reporter.ResultsGallery instance

    hardlink_output : bool, default=True
        If True, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    caching : bool, default=True
        Whether to cache the intermediary results or not. For faster
        data processing. If same step is repeated, it loads directly
        from cache.
    """
    # sanitze subject data
    if isinstance(subject_data, dict):
        subject_data = SubjectData(**subject_data)
    else:
        assert isinstance(subject_data, SubjectData), (
            "subject_datta must be SubjectData instance or dict, "
            "got %s" % type(subject_data))

    # get ready for reporting
    if report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = generate_preproc_undergone_docstring(
            bet=do_bet,
            realign=do_mc,
            coregister=do_coreg,
            normalize=do_normalize,
        )

        # initialize report factory
        subject_data.init_report(parent_results_gallery=parent_results_gallery,
                                 preproc_undergone=preproc_undergone)

    else:
        subject_data._set_session_ids()
        subject_data._sanitize_output_dirs()
        subject_data._sanitize_scratch_dirs()

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        nipype_mem = NipypeMemory(base_dir=cache_dir)
        joblib_mem = JoblibMemory(cache_dir, verbose=100)

    # sanitize input files
    if (isinstance(subject_data.func, list) and
            not isinstance(subject_data.func[0], _basestring)):
        subject_data.func = joblib_mem.cache(do_fsl_merge)(
            subject_data.func, subject_data.output_dir, output_prefix='Merged',
            cmd_prefix=cmd_prefix)

    ######################
    #  Skip dummy scans
    ######################
    if remove_dummy_scans:
        subject_data = _do_subject_extract_roi(subject_data, caching=caching,
                                               cmd_prefix=cmd_prefix,
                                               t_min=n_dummy_scans,
                                               hardlink_output=hardlink_output,
                                               )
    ######################
    #  Brain Extraction
    ######################
    if do_bet:
        subject_data = _do_subject_bet(subject_data, caching=caching,
                                       cmd_prefix=cmd_prefix,
                                       hardlink_output=hardlink_output,
                                       **kwargs.get('fraction', {}))
    #######################
    #  FSL Motion Outliers
    #######################
    if do_fsl_motion_outliers:
        subject_data = _do_subject_fsl_motion_outliers(
            subject_data, caching=caching,
            fsl_motion_outliers_metric=fsl_motion_outliers_metric,
            cmd_prefix=cmd_prefix,
            hardlink_output=hardlink_output,
            **kwargs)

    #######################
    #  Motion correction
    #######################
    if do_mc:
        subject_data = _do_subject_mcflirt(subject_data, caching=caching,
                                           register_to_mean=register_to_mean,
                                           cost=cost,
                                           cmd_prefix=cmd_prefix,
                                           hardlink_output=hardlink_output,
                                           report=report, **kwargs)
    else:
        if do_ica_aroma:
            raise ValueError("ICA Aroma is set to True but you have not "
                             "set MCFLIRT to True. "
                             "Realignment parameters are necessary to do "
                             "ICA Aroma. ")

    ###################
    # Coregistration
    ###################
    if do_coreg:
        subject_data = _do_subject_coregister(subject_data, caching=caching,
                                              cmd_prefix=cmd_prefix,
                                              hardlink_output=hardlink_output,
                                              report=report, **kwargs)
    else:
        if do_ica_aroma:
            raise ValueError("ICA Aroma is set to True but you have not "
                             "set co-registration step to True. "
                             "Coregistration is a required step "
                             "to do ICA Aroma since ICA Aroma uses "
                             "func_to_struct.mat file from co-registration.")
    ##########################
    # Spatial normalization
    ##########################
    if do_normalize:
        subject_data = _do_subject_normalize(subject_data, caching=caching,
                                             cmd_prefix=cmd_prefix,
                                             hardlink_output=hardlink_output,
                                             report=report, do_coreg=do_coreg,
                                             **kwargs)
    else:
        if do_ica_aroma:
            raise ValueError("ICA Aroma is set to True but you have not "
                             "set spatial normalization to True. "
                             "Spatial normalization is a required step "
                             "to do ICA Aroma. ICA Aroma used fnirt warp "
                             "file from spatial normalization.")
    #########
    # Smooth
    #########
    if do_smooth:
        if not fwhm > 0.:
            warn_msg = 'Smoothing is specified but fwhm={0}'
            warnings.warn(warn_msg.format(fwhm), stacklevel=2)
        subject_data = _do_subject_smoothing(subject_data, fwhm=fwhm,
                                             caching=caching, cmd_prefix=cmd_prefix,
                                             hardlink_output=hardlink_output,
                                             report=report, **kwargs)
    ############
    # ICA-Aroma
    ############
    if do_ica_aroma:
        subject_data = _do_subject_ica_aroma(subject_data,
                                             ica_aroma_denoise_type=ica_aroma_denoise_type,
                                             caching=caching,
                                             hardlink_output=hardlink_output,
                                             **kwargs)

    return subject_data
