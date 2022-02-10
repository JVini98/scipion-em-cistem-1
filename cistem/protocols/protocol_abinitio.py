# **************************************************************************
# *
# * Authors:     Federico P. de Isidro Gomez (fp.deisidro@cnb.csic.es) [1]
# * Authors:     Jaime Viniegra de la Fuente [1]
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import re
from glob import glob
from collections import OrderedDict
from pwem.protocols import EMProtocol

from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        IntParam, BooleanParam,
                                        StringParam)
from pyworkflow.utils.path import (makePath, createLink,
                                   cleanPattern, moveFile)
from pyworkflow.object import Float
from pwem.protocols import ProtClassify2D

from cistem import Plugin
from ..convert import (writeReferences, geometryFromMatrix,
                       rowToAlignment, HEADER_COLUMNS)
from ..constants import *


class CistemProtAbinitio(EMProtocol):
    """ Protocol to run Abinitio in cisTEM. """
    _label = 'abinitio'

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)

    def _createFilenameTemplates(self):
        """ Centralize the names of the files. """
        myDict = {
            'run_stack': 'Abinitio/ParticleStacks/particle_stack_%(run)02d.mrc',
            'initial_cls': 'Abinitio/ClassAverages/reference_averages.mrc',
            'iter_cls': 'Abinitio/ClassAverages/class_averages_%(iter)04d.mrc',
            'iter_par': 'Abinitio/Parameters/classification_input_par_%(iter)d.par',
            'iter_par_block': 'Abinitio/Parameters/classification_input_par_%(iter)d_%(block)d.par',
            'iter_cls_block': 'Abinitio/ClassAverages/class_dump_file_%(iter)d_%(block)d.dump',
            'iter_cls_block_seed': 'Abinitio/ClassAverages/class_dump_file_%(iter)d_.dump'
        }
        self._updateFilenamesDict(myDict)

    def _initialize(self):
        self._createFilenameTemplates()

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSetOfParticles', PointerParam, important=True,
                      # condition='not doContinue',
                      label='Input Set of Particles',
                      help='Input set of particles',
                      pointerClass='SetOfParticles')
        form.addParam('numberOfStarts', IntParam, default=2,
                      label='Number of Starts',
                      help='The number of times the ab-initio reconstruction'
                           ' is restarted, using the result from the previous'
                           ' run in each restart.')
        form.addParam('numberOfCyclesPerStart', IntParam, default=40,
                      label='No. of Cycles per Start',
                      help='The number of refinement cycles to run for each start.'
                           ' The percentage of particles and the refinement resolution'
                           ' limit will be adjusted automatically from cycle to cycle'
                           ' using initial and final values specified under Expert Options.')

        form.addSection(label='Expert Options')
        group = form.addGroup('Refinement')
        group.addParam('initialRes', FloatParam, default=20.0,
                       label='Initial Resolution Limit (A)',
                       help='The starting resolution limit used to align particles '
                            'against the current 3D reconstruction. In most cases, '
                            'this should specify a relatively low resolution to make'
                            ' sure the reconstructions generated in the initial '
                            'refinement cycles do not develop spurious high-resolution'
                            ' features.')
        group.addParam('finalRes', FloatParam, default=8.0,
                       label='Final Resolution Limit (A)',
                       help='The resolution limit used in the final refinement cycle.'
                            ' In most cases, this should specify a resolution at which'
                            ' expected secondary structure becomes apparent, i.e. around 9 Å.')
        # En uso prepare stack y refine 3d
        group.addParam('maskRadius', FloatParam, default=90.0,
                       label='Mask Radius (A)')
        # En uso refine 3d
        group.addParam('innerMaskRadius', FloatParam, default=0.0,
                       label='Inner Mask Radius (A)')
        group.addParam('searchRangeX', FloatParam, default=18.0,
                       label='Search Range in X (A)')
        group.addParam('searchRangeY', FloatParam, default=18.0,
                       label='Search Range in Y (A)')
        group.addParam('autoMasking', BooleanParam, default=True,
                       label='Use Auto-Masking?',
                       help='Should the 3D reconstructions be masked? Masking is important'
                            ' to suppress weak density features that usually appear in the'
                            ' early stages of ab-initio reconstruction, thus preventing them'
                            ' to get amplified during the iterative refinement. Masking should'
                            ' only be disabled if it appears to interfere with the reconstruction'
                            ' process.')
        group.addParam('autoPercent', BooleanParam, default=True,
                       label='Auto Percent Used?',
                       help='Should the percentage of particles used in each refinement cycle be'
                            ' set automatically? If reconstructions appear very noisy or reconstructions'
                            ' settle into a wrong structure that does not change anymore during'
                            ' iterations, disable this option and specify initial and final percentages'
                            ' manually. To reduce noise, increase the percentage; to make reconstructions'
                            ' more variable, decrease the percentage. By default, the initial percentage'
                            ' is set to include an equivalent of 2500 asymmetric units and the final'
                            ' percentage corresponds to 10,000 asymmetric units used.')

        group.addParam('initialPercent', FloatParam, default=10.00, condition='not autoPercent',
                       label='Initial % Used',
                       help='User-specified percentages of particles used when Auto Percent Used is disabled.')
        group.addParam('finalPercent', FloatParam, default=10.00, condition='not autoPercent',
                       label='Final % Used',
                       help='User-specified percentages of particles used when Auto Percent Used is disabled.')
        group.addParam('applySymmetry', BooleanParam, default=False,
                       label='Always Apply Symmetry?')

        group = form.addGroup('3D reconstruction')
        # En uso en reconstruct 3d
        group.addParam('likehoodBluring', BooleanParam, default=False,
                       label='Apply Likelihood Blurring?',
                       help='Should the reconstructions be blurred by inserting each particle image at'
                            ' multiple orientations, weighted by a likelihood function? Enable this option'
                            ' if the ab-initio procedure appears to suffer from over-fitting and the appearance'
                            ' of spurious high-resolution features.')
        # En uso en reconstruct 3d
        group.addParam('smoothingFactor', FloatParam, condition='likehoodBluring', default=1.0,
                       label='Smoothing Factor',
                       help='A factor that reduces the range of likelihoods used for blurring. A smaller'
                            ' number leads to more blurring. The user should try values between 0.1 and 1.')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):

        self._createWorkingDirs()
        self._createFilenameTemplates()
        self._insertFunctionStep('prepareStack')
        # for j in range(0, self.numberOfStarts.get()):
        #    print(f"Start: {j}")
        #    for i in range(0, self.numberOfCyclesPerStart.get()):
        #        print(f"Ciclo: {i}")
        self._insertFunctionStep('reconstruct3d')
        self._insertFunctionStep('merge3d')
        #        self._insertFunctionStep('refine3d')

    # --------------------------- STEPS functions -----------------------------


    def prepareStack(self):

        # Utiliza solo hasta el No por defecto
        paramsPrepareStack = {'input_stack': '/home/jvini/Escritorio/AbinitioFiles/particle_stack_1.mrc'#self.inputSetOfParticles.get().getFileName()
            , 'output_stack': self._getExtraPath(self._getFileName('run_stack', run=0))
            , 'pix_Size': self._getInputParticles().getSamplingRate()
            , 'mask_radius': self.maskRadius.get()
                              }
        argsPrepareStack = """ << eof
%(input_stack)s
%(output_stack)s
%(pix_Size)f
%(mask_radius)f
No
eof
        """
        command = argsPrepareStack % paramsPrepareStack

        self.runJob(self._getProgram('prepare_stack'), command,
                    # cwd=self._getExtraPath(),
                    env=Plugin.getEnviron())

    def reconstruct3d(self):

        paramsReconstruct3d = self.getReconstruct3dParams(self.likehoodBluring.get(), self.innerMaskRadius.get(),
                                                          self.maskRadius.get(), self.smoothingFactor.get())
        argsReconstruct3d = """ << eof
%(input_stack)s
%(input_frealign)s
%(input_recons)s
%(output_recons_1)s
%(output_recons_2)s
%(output_filt_recons)s
%(output_res_stats)s
%(part_symmetry)s
%(first_particle)d
%(last_particle)d
%(pixel_size)f
%(beam_energy)f
%(spherical_aber)f
%(amp_contrast)f
%(molecular_mass)f
%(inner_mask_rad)f
%(outer_mask_rad)f
%(rec_resolution_limit)f
%(ref_resolution_limit)f
%(part_weighting_factor)f
%(score_threshold)f
%(smoothing_factor)f
%(padding_factor)f
%(normalize_particles)s
%(adjust_scores)s
%(invert_particle_contrast)s
%(exclude_images)s
%(crop_particle_images)s
%(FSCc_calculation)s
%(center_mass)s
%(likelihood_blurring)s
%(threshold_input_reconstruction)s
%(dump_intermediate_arrays)s
%(dump_filename_odd)s
%(dump_filename_even)s
eof
"""
        command = argsReconstruct3d % paramsReconstruct3d

        self.runJob(self._getProgram('reconstruct3d'), command,
                    # cwd=self._getExtraPath(),
                    env=Plugin.getEnviron())

    def merge3d(self):
        paramsMerge3d = self.getMerge3dParams()
        argsMerge3d = """ << eof
%(output_rec_1)s
%(output_rec_2)s			
%(output_filt_rec)s			
%(output_res_stats)s		
%(molecular_mass_of_particle)f			
%(inner_mask_radius)f					
%(outer_mask_radius)f					
%(input_dump_odd_part)s	
%(input_dump_even_part)s	
%(class_number)d	
%(save_orthogonal_views_img)s
%(orthogonal_views_filename)s
%(num_of_dump_files)d		
%(weinner_nominator)f	
eof
        """
        command = argsMerge3d % paramsMerge3d

        self.runJob(self._getProgram('merge3d'), command,
                    cwd=self._getExtraPath(),
                    env=Plugin.getEnviron())

    def refine3d(self):
        paramsRefine3d = self.getRefine3dParams()
        argsRefine3d = """ << eof
%(input_particle_img)s			
%(input_frealign_param_filename)s	
%(input_rec)s		
%(input_data_stats)s			
%(use_stats)s		
%(output_matching_projections)s		
%(output_param_file)s	
%(output_param_changes)s		
%(particle_symmetry)s		
%(first_particle_to_refine)d 	
%(last_particle_to_refine)d 		
%(percent_of_particles_to_use)f
%(pixel_size_of_imgs)f		
%(beam_energy)f			
%(spherical_aberration)f		
%(amplitude_contrast)f			
%(molecular_mass_of_particle)f		
%(inner_mask_radius)f		
%(outer_mask_radius)f			
%(low_res_limit)f	
%(high_res_limit)f	
%(res_limit_for_signed_CC)f	
%(res_limit_for_classification)f	
%(mask_radius_for_global_search)f	
%(approx.res_limit_for_search)f	
%(angular_step)f		
%(num_of_top_hits_to_refine)d
%(search_range_in_X)f	
%(search_range_in_Y)f		
%(2D_mask_x_coordinate)f	
%(2D_mask_y_coordinate)f
%(2D_mask_z_coordinate)f	
%(2D_mask_radius)f		
%(defocus_search_range)f	
%(defocus_step)f		
%(tuning_params_padding_factor)f	
%(global_search)s		
%(local_refinement)s		
%(refine_psi)s		
%(refine_theta)s		
%(refine_phi)s	
%(refine_shiftX)s		
%(refine_shiftY)s
%(calculate_matching_projections)s
%(apply_2D_masking)s		
%(refine_defocus)s
%(normalize_particles)s	
%(invert_particle_contrast)s	
%(exclude_imgs_with_blank_edges)s	
%(normalize_input_rec)s	
%(threshold_input_rec)s	
%(local_global_refine)s		
%(current_class)d		
%(ignore_input_angles)s	
eof
    """
        command = argsRefine3d % paramsRefine3d

        self.runJob(self._getProgram('refine3d'), command,
                    cwd=self._getExtraPath(),
                    env=Plugin.getEnviron())

    # --------------------------- INFO functions ------------------------------

    # def _validate(self):
    #     errors = []
    #
    #     if self.doContinue:
    #         continueProtocol = self.continueRun.get()
    #         if (continueProtocol is not None and
    #                 continueProtocol.getObjId() == self.getObjId()):
    #
    #             errors.append('In Scipion you must create a new cisTEM run')
    #             errors.append('and select the continue option rather than')
    #             errors.append('select continue from the same run.')
    #             errors.append('')  # add a new line
    #         errors += self._validateContinue()
    #
    #     return errors

    def _summary(self):
        self._initialize()
        lastIter = self._lastIter()

        if lastIter is not None:
            iterMsg = 'Iteration %d' % lastIter
            if self.hasAttribute('numberOfIterations'):
                iterMsg += '/%d' % self._getnumberOfIters()
        else:
            iterMsg = 'No iterations finished yet.'

        summary = [iterMsg]

        summary.append("Input Particles: %s" % self.getObjectTag('inputParticles'))
        summary += self._summaryNormal()
        return summary

    def _methods(self):
        methods = "We classified input particles %s (%d items) " % (
            self.getObjectTag('inputParticles'),
            self._getPtclsNumber())
        methods += "into %d classes using refine2d" % self.numberOfClassAvg
        return [methods]

    # --------------------------- UTILS functions -----------------------------
    def _createWorkingDirs(self):
        for dirFn in ['Abinitio/ParticleStacks',
                      'Abinitio/ClassAverages',
                      'Abinitio/Parameters']:
            makePath(self._getExtraPath(dirFn))

    def _getInputParticles(self):
        return self.inputSetOfParticles.get()

    def _getProgram(self, program):

        return Plugin.getProgram(program)

    def getReconstruct3dParams(self, likehoodBluring, inMaskRadius, outMaskRadius, smoothingFactor, run):

        paramsReconstruct3d = {
            'input_stack': self._getExtraPath(self._getFileName('run_stack', run)),
            'input_frealign': "/home/parallels/ScipionUserData/projects/abinitio/Runs/000507_CistemProtRefine2D/extra/Refine2D/Parameters/classification_input_par_1.par",
            'input_recons': "/dev/null",
            'output_recons_1': "/dev/null",  # self._getExtraPath("my_reconstruction_1.mrc"),
            'output_recons_2': "/dev/null",  # self._getExtraPath("my_reconstruction_2.mrc"),
            'output_filt_recons': "/dev/null",  # self._getExtraPath("my_filtered_reconstruction.mrc"),
            'output_res_stats': "/dev/null",  # self._getExtraPath("my_statistics.txt"),
            'part_symmetry': "C1",
            'first_particle': 1,
            'last_particle': 0,
            'pixel_size': 1.5,
            'beam_energy': self._getInputParticles().getAcquisition().getVoltage(),
            'spherical_aber': 2.7,
            'amp_contrast': self._getInputParticles().getAcquisition().getAmplitudeContrast(),
            'molecular_mass': 1000,
            'inner_mask_rad': inMaskRadius,
            'outer_mask_rad': outMaskRadius,
            'rec_resolution_limit': 0,
            'ref_resolution_limit': 0,
            'part_weighting_factor': 5,
            'score_threshold': 1,
            'smoothing_factor': smoothingFactor,
            'padding_factor': 1,
            'normalize_particles': "No",
            'adjust_scores': "Yes",
            'invert_particle_contrast': "No",
            'exclude_images': "No",
            'crop_particle_images': "No",
            'FSCc_calculation': "No",
            'center_mass': "Yes",
            'likelihood_blurring': "Yes",
            'threshold_input_reconstruction': "No",
            'dump_intermediate_arrays': "Yes",
            'dump_filename_odd': self._getExtraPath("dump_file_1.dat"),
            'dump_filename_even': self._getExtraPath("dump_file_2.dat"),
        }

        if likehoodBluring:
            paramsReconstruct3d['likelihood_blurring'] = "Yes"
        else:
            paramsReconstruct3d['likelihood_blurring'] = "No"

        return paramsReconstruct3d

    def getMerge3dParams(self):

        paramsMerge3d = {
            'output_rec_1': "my_reconstruction_1.mrc",
            'output_rec_2': "my_reconstruction_2.mrc",
            'output_filt_rec': "my_filtered_reconstruction.mrc",
            'output_res_stats': "my_statistics.txt",
            'molecular_mass_of_particle': 440.00,
            'inner_mask_radius': 0.10,
            'outer_mask_radius': 90.00,
            # 'input_dump_odd_part': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Scratch/Startup/startup_dump_file_0_odd_.dump",
            # 'input_dump_even_part': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Scratch/Startup/startup_dump_file_0_even_.dump",
            'input_dump_odd_part': self._getExtraPath("dump_file_1.dump"),
            'input_dump_even_part': self._getExtraPath("dump_file_2.dump"),
            'class_number': 1,
            'save_orthogonal_views_img': "No",
            'orthogonal_views_filename': "",
            'num_of_dump_files': 2,
            'weinner_nominator': 1
        }

        return paramsMerge3d

    def getRefine3dParams(self):

        paramsRefine3d = {
            'input_particle_img': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Assets/ParticleStacks/particle_stack_0.mrc",
            'input_frealign_param_filename': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Assets/Parameters/startup_input_par_0_1.par",
            'input_rec': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Assets/Volumes/startup_volume_1_1.mrc",
            'input_data_stats': "/home/parallels/Desktop/Cistem_pruebasAna/test1/Assets/Parameters/startup_input_stats_0_1.txt",
            'use_stats': "Yes",
            'output_matching_projections': "my_projection_stack.mrc",
            'output_param_file': "my_refined_parameters.par",
            'output_param_changes': "my_parameter_changes.par",
            'particle_symmetry': "C1",
            'first_particle_to_refine': 1,
            'last_particle_to_refine': 0,
            'percent_of_particles_to_use': 1.00,
            'pixel_size_of_imgs': 2.00,
            'beam_energy': self._getInputParticles().getAcquisition().getVoltage(),
            'spherical_aberration': self._getInputParticles().getAcquisition().getSphericalAberration(),
            'amplitude_contrast': self._getInputParticles().getAcquisition().getAmplitudeContrast(),
            'molecular_mass_of_particle': 1000.00,
            'inner_mask_radius': 0.00,
            'outer_mask_radius': 100.00,
            'low_res_limit': 330.00,
            'high_res_limit': 8.00,
            'res_limit_for_signed_CC': 0.20,
            'res_limit_for_classification': 0.30,
            'mask_radius_for_global_search': 98.00,
            'approx.res_limit_for_search': 20.00,
            'angular_step': 0.40,
            'num_of_top_hits_to_refine': 19,
            'search_range_in_X': 0.50,
            'search_range_in_Y': 0.60,
            '2D_mask_x_coordinate': 97.00,
            '2D_mask_y_coordinate': 96.00,
            '2D_mask_z_coordinate': 95.00,
            '2D_mask_radius': 94.00,
            'defocus_search_range': 500.00,
            'defocus_step': 50.00,
            'tuning_params_padding_factor': 3.00,
            'global_search': "Yes",
            'local_refinement': "No",
            'refine_psi': "Yes",
            'refine_theta': "No",
            'refine_phi': "Yes",
            'refine_shiftX': "No",
            'refine_shiftY': "Yes",
            'calculate_matching_projections': "No",
            'apply_2D_masking': "Yes",
            'refine_defocus': "No",
            'normalize_particles': "Yes",
            'invert_particle_contrast': "No",
            'exclude_imgs_with_blank_edges': "Yes",
            'normalize_input_rec': "No",
            'threshold_input_rec': "Yes",
            'local_global_refine': "No",
            'current_class': 0,
            'ignore_input_angles': "No"}
        return paramsRefine3d