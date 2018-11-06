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
        in_file=subject_data.func[0], t_min=t_min, t_size=-1,
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
        in_file=subject_data.anat, interface_kwargs=kwargs))

    # failed node
    if bet_results.outputs is None:
        subject_data.failed = True
        _logger.error(_INTERFACE_ERROR_MSG.format(
            bet_results.interface, bet_results.version,
            bet_results.inputs, bet_results.runtime.traceback))
        return subject_data

    # collect output
    subject_data.anat = bet_results.outputs.out_file

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

    # Simple dot product between func2structmat and struct2MNImat
    func2structmat = subject_data.nipype_results['coreg'].outputs.out_matrix_file
    affine_file = norm_results.outputs.out_matrix_file
    premat = np.dot(np.loadtxt(func2structmat), np.loadtxt(affine_file))
    np.savetxt('this_affine.mat', premat)
    applywarp_results = applywarp(**_update_interface_inputs(
        ref_file=FSL_WARP_TEMPLATE, in_file=subject_data.func[0],
        field_file=fnirt_results.outputs.fieldcoeff_file,
        premat='this_affine.mat', interface_kwargs=kwargs))
    subject_data.nipype_results['applywarp'] = applywarp_results
    subject_data.func = applywarp_results.outputs.out_file

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    if report:
        subject_data.generate_normalization_thumbnails()
    return subject_data.sanitize()


def do_subject_preproc(subject_data,
                       do_bet=True,
                       do_mc=True,
                       register_to_mean=True,
                       do_coreg=True,
                       do_normalize=True,
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

    if caching:
        # prepare for smart-caching
        cache_dir = os.path.join(subject_data.scratch, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        nipype_mem = NipypeMemory(base_dir=cache_dir)
        joblib_mem = JoblibMemory(cache_dir, verbose=100)

    # sanitize input files
    if not isinstance(subject_data.func[0], _basestring):
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
    #  Motion correction
    #######################
    if do_mc:
        subject_data = _do_subject_mcflirt(subject_data, caching=caching,
                                           register_to_mean=register_to_mean,
                                           cmd_prefix=cmd_prefix,
                                           hardlink_output=hardlink_output,
                                           report=report, **kwargs)
    ###################
    # Coregistration
    ###################
    if do_coreg:
        subject_data = _do_subject_coregister(subject_data, caching=caching,
                                              cmd_prefix=cmd_prefix,
                                              hardlink_output=hardlink_output,
                                              report=report, **kwargs)
    ##########################
    # Spatial normalization
    ##########################
    if do_normalize:
        subject_data = _do_subject_normalize(subject_data, caching=caching,
                                             cmd_prefix=cmd_prefix,
                                             hardlink_output=hardlink_output,
                                             report=report, do_coreg=do_coreg,
                                             **kwargs)

    return subject_data
