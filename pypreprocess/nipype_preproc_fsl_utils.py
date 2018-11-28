"""
Author: Bertrand Thirion, Alexandre Abraham, DOHMATOB Elvis Dopgima

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
        if not ext is None:
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


def _do_subject_extract_roi(subject_data, caching, cmd_prefix,
                            t_min, hardlink_output, output_prefix='extract_',
                            **kwargs):
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin dummy scans "
                      "removal step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data
    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        extract = subject_data.mem.cache(fsl.ExtractROI)
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
        _logger.error(_INTERFACE_ERROR_MSG.format(
            extract_roi_results.interface, extract_roi_results.version,
            extract_roi_results.inputs, extract_roi_results.runtime.traceback))
        return subject_data

    # collect output
    subject_data.func = extract_roi_results.outputs.roi_file

    subject_data.nipype_results['dummy scans'] = extract_roi_results

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_bet(subject_data, caching, cmd_prefix,
                    hardlink_output, report, **kwargs):
    """
    """
    if not subject_data.anat:
        warnings.warn("subject_data.anat=%s (empty); skippin Brain Extraction"
                      % (subject_data.anat), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        bet = subject_data.mem.cache(fsl.BET)
    else:
        bet = fsl.BET().run

    if not fsl.BET._cmd.startswith("fsl"):
        fsl.BET._cmd = cmd_prefix + fsl.BET._cmd

    bet_results = bet(**_update_interface_inputs(
        in_file=subject_data.anat, mask=True, interface_kwargs=kwargs))

    # failed node
    if bet_results.outputs is None:
        subject_data.failed = True
        _logger.error(_INTERFACE_ERROR_MSG.format(
            bet_results.interface, bet_results.version,
            bet_results.inputs, bet_results.runtime.traceback))
        return subject_data

    # collect output
    subject_data.anat = bet_results.outputs.out_file
    subject_data.anat_mask = bet_results.outputs.mask_file

    subject_data.nipype_results['bet'] = bet_results

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_mcflirt(subject_data, caching, register_to_mean,
                        cmd_prefix, hardlink_output, report, **kwargs):
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin mcflirt "
                      "realignment step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        mcflirt = subject_data.mem.cache(fsl.MCFLIRT)
    else:
        mcflirt = fsl.MCFLIRT().run

    if not fsl.MCFLIRT._cmd.startswith("fsl"):
        fsl.MCFLIRT._cmd = cmd_prefix + fsl.MCFLIRT._cmd

    mcflirt_results = mcflirt(**_update_interface_inputs(
        in_file=subject_data.func[0], mean_vol=register_to_mean,
        cost='mutualinfo', save_mats=True, save_plots=True,
        interface_kwargs=kwargs))

    # failed node
    if mcflirt_results.outputs is None:
        subject_data.failed = True
        _logger.error(_INTERFACE_ERROR_MSG.format(
            mcflirt_results.interface, mcflirt_results.version,
            mcflirt_results.inputs, mcflirt_results.runtime.traceback))
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
                                    cmd_prefix, hardlink_output, report,
                                    **kwargs):
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin compute motion "
                      "outliers step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        motion_outliers = subject_data.mem.cache(utils.MotionOutliers)
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
        _logger.error(_INTERFACE_ERROR_MSG.format(
            motion_outliers_results.interface, motion_outliers_results.version,
            motion_outliers_results.inputs, motion_outliers_results.runtime.traceback))
        return subject_data

    # collect output
    subject_data.motion_outliers = motion_outliers_results.outputs.out_file
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
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin flirt "
                      "coregistration step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        flirt = subject_data.mem.cache(fsl.FLIRT)
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
        _logger.error(_INTERFACE_ERROR_MSG.format(
            coreg_results.interface, coreg_results.version,
            coreg_results.inputs, coreg_results.runtime.traceback))
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
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin fnirt and "
                      "applywarp based normalization step "
                      % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        flirt = subject_data.mem.cache(fsl.FLIRT)
    else:
        flirt = fsl.FLIRT().run

    if not fsl.FLIRT._cmd.startswith("fsl"):
        fsl.FLIRT._cmd = cmd_prefix + fsl.FLIRT._cmd

    norm_results = flirt(**_update_interface_inputs(
        in_file=subject_data.anat, reference=FSL_T1_TEMPLATE,
        interface_kwargs=kwargs))

    # failed node
    if norm_results.outputs is None:
        subject_data.failed = True
        _logger.error(_INTERFACE_ERROR_MSG.format(
            norm_results.interface, norm_results.version,
            norm_results.inputs, norm_results.runtime.traceback))
        return subject_data

    subject_data.nipype_results['flirt2'] = norm_results
    # collect output
    subject_data.anat = norm_results.outputs.out_file

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        fnirt = subject_data.mem.cache(fsl.FNIRT)
    else:
        fnirt = fsl.FNIRT().run

    if not fsl.FNIRT._cmd.startswith("fsl"):
        fsl.FNIRT._cmd = cmd_prefix + fsl.FNIRT._cmd
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
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        applywarp = subject_data.mem.cache(fsl.ApplyWarp)
    else:
        applywarp = fsl.ApplyWarp().run

    if not fsl.ApplyWarp._cmd.startswith("fsl"):
        fsl.ApplyWarp._cmd = cmd_prefix + fsl.ApplyWarp._cmd

    # Apply final transform
    func2structmat = subject_data.nipype_results['coreg'].outputs.out_matrix_file
    applywarp_results = applywarp(**_update_interface_inputs(
        ref_file=FSL_WARP_TEMPLATE,
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
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skippin smoothing "
                      "step " % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        smooth = subject_data.mem.cache(fsl.Smooth)
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
        _logger.error(_INTERFACE_ERROR_MSG.format(
            smooth_results.interface, smooth_results.version,
            smooth_results.inputs, smooth_results.runtime.traceback))
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
    """
    """
    if not subject_data.func[0]:
        warnings.warn("subject_data.func=%s (empty); skipping ICA-AROMA "
                      "step " % (subject_data.func[0]), stacklevel=2)
        return subject_data

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        ica_aroma = subject_data.mem.cache(fsl.ICA_AROMA)
    else:
        ica_aroma = fsl.ICA_AROMA().run

    func2structmat = subject_data.nipype_results['coreg'].outputs.out_matrix_file
    ica_aroma_results = ica_aroma(**_update_interface_inputs(
        in_file=subject_data.func, mat_file=func2structmat,
        fnirt_warp_file=fnirt_results.outputs.field_file,
        motion_parameters=subject_data.realignment_parameters[0],
        denoise_type=ica_aroma_denoise_type, interface_kwargs=kwargs))

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_normalization_thumbnails()
    return subject_data.sanitize()


def do_subject_preproc(subject_data,
                       do_bet=True,
                       do_mc=True,
                       do_fsl_motion_outliers=True,
                       fsl_motion_outliers_metric='fdrms',
                       register_to_mean=True,
                       do_coreg=True,
                       do_normalize=True,
                       do_smooth=True,
                       do_ica_aroma=False,
                       ica_aroma_denoise_type='nonaggr',
                       fwhm=0.,
                       remove_dummy_scans=True,
                       n_dummy_scans=5,
                       tsdiffana=True,
                       cmd_prefix="fsl5.0-",
                       report=True,
                       parent_results_gallery=None,
                       preproc_undergone="",
                       hardlink_output=True,
                       caching=True,
                       **kwargs
                       ):
    """
    Preprocesses subject data using FSL.

    Parameters
    ----------

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
                                 preproc_undergone=preproc_undergone,
                                 tsdiffana=tsdiffana)

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
                                       report=False,
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
            report=report, **kwargs)

    #######################
    #  Motion correction
    #######################
    if do_mc:
        subject_data = _do_subject_mcflirt(subject_data, caching=caching,
                                           register_to_mean=register_to_mean,
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
