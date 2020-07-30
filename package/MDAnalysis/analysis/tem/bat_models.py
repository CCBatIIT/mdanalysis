# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
r"""BAT Ensemble Models --- :mod:`MDAnalysis.analysis.tem.bat_models`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains classes that model thermodynamic ensembles based on their
Bond-Angle-Torsion (BAT) coordinates.


See Also
--------
:func:`~MDAnalysis.analysis.encore.similarity.hes()`
    function that compares two ensembles after representing each as a harmonic
    oscillator


Example applications
--------------------

The :class:`~MDAnalysis.analysis.bat.IndependentGaussianModel` class models
the bond lengths, angles, and torsions as independent Gaussians and external
coordinates by a kernel density estimate. For example, we can create ensemble
models for residues 5-10 of adenylate kinase (AdK). The trajectory is
included within the test data files::

   import MDAnalysis as mda
   from MDAnalysis.analysis.tem.bat_models import IndependentGaussianModel

   from MDAnalysisTests.datafiles import PSF, DCD
   import numpy as np

   u = mda.Universe(PSF, DCD)

   # selection of atomgroups
   selected_residues = u.select_atoms("resid 5-10")

   # Initiate the ensemble model, including calculating BAT coordinates
   enmodel = IndependentBATEnsembleModel(selected_residues)

   # This attribute contains the log partition function for the model
   print(enmodel.lnZ)

   # Generate 10 random samples
   print(enmodel.rvs(10))

   # Evaluate the log probability density of the initial configuration
   print(enmodel._())


References
----------

.. [Minh2020] Minh, David D L (2020). "Alchemical Grid Dock (AlGDock): Binding
   Free Energy Calculations between Flexible Ligands and Rigid Receptors."
   *Journal of Computational Chemistry* 41(7): 715–30.
   doi:`10.1002/jcc.26036 <https://doi.org/10.1002/jcc.26036>`_

.. [Gyimesi2017] Gyimesi, Gergely, Péter Závodszky, and András Szilágyi.
   “Calculation of Configurational Entropy Differences from Conformational
   Ensembles Using Gaussian Mixtures.” *Journal of Chemical Theory and
   Computation* 13(1): 29–41. `doi:10.1021/acs.jctc.6b00837
   <https://doi.org/10.1021/acs.jctc.6b00837>`_

"""
from __future__ import absolute_import
# from six.moves import zip, range

import numpy as np

import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.bat import BAT

from .partial_models.independentgaussian import IndependentGaussian
from .partial_models.multivariategaussian import MultivariateGaussian
from .partial_models.gaussiankde import IndependentKDE, PrincipalComponentsKDE
from .partial_models.gaussianmixture import GaussianMixture
from .partial_models.external import Translational, Rotational

tau = 2 * np.pi


def measure_torsion_shifts(A):
    """ Measure the angle to shift torsions

    Shifting torsions so that the density is minimal at the periodic boundary
    is useful for describing data with a nonperiodic distribution function.
    To measure this shift, a histogram is generated for every degree of freedom.
    The longest interval where the number of counts is minimal is identified.
    The shift is the angle that should be subtracted so that it occurs at pi.
    This function was adapted from [Gyimesi20XX]_.

    Parameters
    ----------
    A : numpy.ndarray
        an array with dimensions (N, K), where N is the number of samples
        and K is the number of degrees of freedom.

    Returns
    -------
    shifts : numpy.ndarray
        an array with dimensions (K,), where K is the number of degrees of freedom.

    """
    shifts = np.zeros(A.shape[1])

    bins = np.linspace(-np.pi, np.pi, 181)
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2.
    bin_centers = np.hstack((bin_centers, bin_centers + 2 * np.pi))

    for k in range(A.shape[1]):
        H = np.histogram(A[:, k], bins=bins)[0]

        # Indices in thte histogram that have the minimum value
        min_inds = (H == np.min(H)).nonzero()[0]
        # duplicate the interval so that the beginning and the end get merged
        min_inds = np.hstack((min_inds, min_inds + len(bins) - 1))

        interval_min = min(min_inds)
        interval_max = interval_min

        largest_interval_length = 0
        largest_interval_min = interval_min
        largest_interval_max = interval_min

        for n in range(1, len(min_inds)):
            if min_inds[n] != (min_inds[n - 1] + 1):
                interval_max = min_inds[n - 1]
                interval_length = interval_max - interval_min
                if interval_length > largest_interval_length:
                    largest_interval_length = interval_max - interval_min
                    largest_interval_min = interval_min
                    largest_interval_max = interval_max
                interval_min = min_inds[n]

        shifts[k] = ((bin_centers[largest_interval_max] + \
                      bin_centers[largest_interval_min])/2. + np.pi)%tau

    return shifts

def split_dofs(bat, n_torsions, torsion_shifts):
    """Split a coordinate array into separate arrays

    Parameters
    ----------
    bat : list of numpy.ndarray
        Each element in the list is an array with dimensions (N, 3A),
        where N is the number of samples and A is the number of atoms.
    n_torsions : list
        The number of torsion angles in each n_molecules
    torsion_shifts : list of numpy.ndarray
        The amount to shift each torsion angle in each molecule

    Returns
    -------
    external : numpy.ndarray
        The external degrees of freedom, translation and rotation.
        An array with dimensions (N, 6), where N is the number of samples.
    bonds : numpy.ndarray
        Bond lengths, starting with r01 and r12, the distances between the
        root atoms. An array with dimensions (N, A-1), where A is the number
        of atoms.
    angles : numpy.ndarray
        Bond angles, starting with a0123, the angle between the root atoms.
        An array with dimensions (N, A-2), where A is the number of atoms.
    shifted_torsions : numpy.ndarray
        Torsion angles, shifted so minimize probability density at the
        periodic boundary. An array with dimensions (N, A-3), where
        A is the number of atoms.
    """

    external = []
    bonds = []
    angles = []
    shifted_torsions = []

    for n in range(len(bat)):
        external.append(bat[n][:, :6])
        bond_indices = [6, 7] + list(range(9, n_torsions[n] + 9))
        angle_indices = [8] + list(range(n_torsions[n] + 9, \
          2 * n_torsions[n] + 9))
        bonds.append(bat[n][:, bond_indices])
        angles.append(bat[n][:, angle_indices])
        st = np.copy(bat[n][:, 2 * n_torsions[n] + 9:])
        st -= torsion_shifts[n]
        st[st < -np.pi] += tau
        shifted_torsions.append(st)

    return (external, bonds, angles, shifted_torsions)

class ThermodynamicEnsembleModel:
    """Create and manage a thermodynamic ensemble model

    Attributes
    ----------
    lnZ : dict
        Log configurational integral/normalizing constant for different
        partial models
    _partial_models : dict
        Partial models used by subclasses to implement rvs and logpdf

    """

    _param_keys = ['model_external', 'coupled_dofs', 'coupling_model', \
                   'n_torsions', 'torsion_shifts', 'reference_external', \
                   'partial_models']

    def __init__(self, model_external, coupled_dofs, coupling_model, \
                 n_torsions, torsion_shifts, reference_external, \
                 partial_models, **kwargs):
        r"""Parameters
        ----------
        model_external : list of bool
            For each molecule, whether to model external degrees of freedom or not.
            If model is loaded from a file, this parameter is ignored.
        coupled_dofs : str
            The degrees of freedom that will be considered to be coupled,
            which is either `angles_torsions` or `torsions`.
            If model is loaded from a file, this parameter is ignored.
        coupling_model : str
            The model for how coupled degrees of freedom are treated, either as
            `IndependentGaussian`, `IndependentKDE`, `MultivariateGaussian`,
            `PrincipalComponentsKDE`, or `GaussianMixture`.
            If model is loaded from a file, this parameter is ignored.
        n_torsions : list
            The number of torsion angles in each n_molecules
        torsion_shifts : list of numpy.ndarray
            The amount to shift each torsion angle in each molecule
        reference_external : list of numpy.ndarray
            Default external degrees of freedom for each molecule
        partial_models : dict
            Dictionary of partial models for subsets of the degrees of freedom
        """
        if coupled_dofs not in ['angle_torsion', 'torsion']:
            raise ValueError('Coupling procedure not supported')
        if coupling_model not in ['IndependentGaussian', 'IndependentKDE', \
                'MultivariateGaussian', 'PrincipalComponentsKDE', \
                'GaussianMixture']:
            raise ValueError('Coupling model not supported')
        if (coupled_dofs == 'angle_torsion') and \
           (coupling_model == 'PrincipalComponentsKDE'):
            raise ValueError('Coupling model incompatible with selected ' + \
                'degrees of freedom')

        self._model_external = model_external
        self._coupled_dofs = coupled_dofs
        self._coupling_model = coupling_model

        self._n_torsions = n_torsions
        self._torsion_shifts = torsion_shifts
        self._reference_external = reference_external

        self._partial_models = partial_models

        inds = np.cumsum([0] + self._n_torsions)
        if coupled_dofs == 'angle_torsion':
            self._angle_inds = inds
            self._torsion_inds = inds + np.sum(self._n_torsions)
        elif coupled_dofs == 'torsion':
            self._torsion_inds = inds

        # Sets up partial models and extracts their partition functions
        if not hasattr(self, '_partial_models') and self._bat is not None:
            self.lnZ = self._extract_lnZ()

    @classmethod
    def from_data(cls, \
          bat, \
          model_external=[False], \
          coupled_dofs = 'torsion', \
          coupling_model = 'IndependentGaussian', \
          source_model=None, **kwargs):
        r"""Build thermodynamic ensemble models based on data

        Independent Gaussian models are used for the bond lengths and angles.
        Different models are used for torsions. External degrees of
        freedom are based on kernel density estimates.

        Parameters
        ----------
        bat : list of numpy.ndarray or None
            Each element in the list is are bat coordinates for a molecule.
            bat coordinates are arrays with dimensions (N, 3A),
            where A is the number of atoms in the molecule.
            The columns are ordered with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.
        model_external : list of bool
            For each molecule, whether to model external degrees of freedom or not.
            If model is loaded from a file, this parameter is ignored.
        coupled_dofs : str
            The degrees of freedom that will be considered to be coupled,
            which is either `angles_torsions` or `torsions`.
            If model is loaded from a file, this parameter is ignored.
        coupling_model : str
            The model for how coupled degrees of freedom are treated, either as
            `IndependentGaussian`, `IndependentKDE`, `MultivariateGaussian`,
            `PrincipalComponentsKDE`, or `GaussianMixture`.
            If model is loaded from a file, this parameter is ignored.
        """
        if len(bat) != len(model_external):
            raise ValueError(
                'Length of bat and model_external lists incompatible')

        n_torsions = [int((b.shape[1] - 9) / 3) for b in bat]
        torsion_shifts = [measure_torsion_shifts(\
            b[:, 2 * n_torsions[-1] + 9:]) for b in bat]
        reference_external = [np.copy(b[0][:6]) for b in bat]

        (external, bonds, angles, shifted_torsions) = \
            split_dofs(bat, n_torsions, torsion_shifts)

        partial_models = {}
        if source_model is None:
            # Define the partial models based on the sampled data
            for n in range(len(bat)):
                if model_external[n]:
                    partial_models[f'translation_{n}'] = \
                        Translational.from_data('translation', \
                            external[n][:,:3])
                    partial_models[f'rotation_{n}'] = \
                        Rotational.from_data(external[n][:,3:6])

                partial_models[f'bonds_{n}'] = \
                    IndependentGaussian.from_data('bond', bonds[n])

                if coupled_dofs == 'torsion':
                    partial_models[f'angles_{n}'] = \
                        IndependentGaussian.from_data('angle', angles[n])
        else:
            # Reuse partial models from source model
            for key in source_model._partial_models.keys():
                partial_models[key] = \
                    source_model._partial_models[key]

        if coupled_dofs == 'angle_torsion':
            coupled = np.hstack(angles + shifted_torsions)
        elif coupled_dofs == 'torsion':
            coupled = np.hstack(shifted_torsions)

        if coupling_model == 'IndependentGaussian':
            partial_models['coupled'] = \
                IndependentGaussian.from_data(coupled_dofs, coupled)
        elif coupling_model == 'IndependentKDE':
            partial_models['coupled'] = \
                IndependentKDE.from_data(coupled_dofs, coupled)
        elif coupling_model == 'MultivariateGaussian':
            partial_models['coupled'] = \
                MultivariateGaussian.from_data(coupled_dofs, coupled)
        elif coupling_model == 'PrincipalComponentsKDE':
            partial_models['coupled'] = \
                PrincipalComponentsKDE.from_data(coupled_dofs, coupled)
        elif coupling_model == 'GaussianMixture':
            partial_models['coupled'] = \
                GaussianMixture.from_data(coupled_dofs, coupled)
        else:
            raise ValueError('Selected coupling model not found')

        return cls(model_external, coupled_dofs, coupling_model, \
             n_torsions, torsion_shifts, reference_external, \
             partial_models)

    @classmethod
    def from_dict(cls, param_dict):
        """Initialize class based on a dictionary of parameters

        Parameters
        ----------
        param_dict : dict
            the dictionary of parameters
        """
        for key in cls._param_keys:
            if not key in param_dict.keys():
                raise ValueError(f'Parameter dictionary missing {key}')

        partial_models = {}
        for key, pm_params in param_dict['partial_model_params']:
          if pm_params['class'].find('IndependentGaussian')>-1:
              partial_models[key] = IndependentGaussian.from_dict(pm_params)
          elif pm_params['class'].find('MultivariateGaussian')>-1:
              partial_models[key] = MultivariateGaussian.from_dict(pm_params)
          elif pm_params['class'].find('IndependentKDE')>-1:
              partial_models[key] = IndependentKDE.from_dict(pm_params)
          elif pm_params['class'].find('PrincipalComponentsKDE')>-1:
              partial_models[key] = PrincipalComponentsKDE.from_dict(pm_params)
          elif pm_params['class'].find('GaussianMixture')>-1:
              partial_models[key] = GaussianMixture.from_dict(pm_params)
          elif pm_params['class'].find('Translational')>-1:
              partial_models[key] = Translational.from_dict(pm_params)
          elif pm_params['class'].find('Rotational')>-1:
              partial_models[key] = Rotational.from_dict(pm_params)
          else:
              raise ValueError('Partial model {0} unknown'.format(params['class']))
        param_dict['partial_models'] = partial_models

        return cls(**{ key:value for (key,value) in param_dict.items() \
            if key in cls._param_keys })

    def to_dict(self):
        """Saves model parameters in a gzipped pickle file

        Parameters
        ----------
        filename : str
            name of the file to save
        """
        param_dict = {'class':repr(getattr(self, '__class__'))}
        for key in self._param_keys:
            if hasattr(self, key):
                param_dict[key] = getattr(self, key)
            elif hasattr(self, '_' + key):
                param_dict[key] = getattr(self, '_' + key)
        param_dict['partial_model_params'] = [\
          (model_name, self._partial_models[model_name].to_dict()) \
              for model_name in self._partial_models.keys()]
        return param_dict

    def rvs(self, N):
        """Generate random samples

        Parameters
        ----------
        N : int
            number of samples to generate

        Returns
        -------
        bat : numpy.ndarray
            an array with dimensions (N,3A), where A is the number of atoms.
            The columns are ordered with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.

        """
        shifted_torsions = self._partial_models['shifted_torsions'].rvs(N)

        external = [np.hstack([\
                self._partial_models[f'translation_{n}'].rvs(N), \
                self._partial_models[f'rotation_{n}'].rvs(N) \
            ]) \
            if self._model_external[n] else None \
            for n in range(len(self._model_external))]

        bonds = [self._partial_models[f'bonds_{n}'].rvs(N) \
            for n in range(len(self._model_external))]

        coupled = self._partial_models['coupled'].rvs(N)
        if self._coupled_dofs == 'torsion':
            angles = [self._partial_models[f'angles_{n}'].rvs(N) \
                for n in range(len(self._model_external))]
        elif self._coupled_dofs == 'angle_torsion':
            angles = [coupled[:,self._angle_inds[n]:self._angle_inds[n+1]] \
                for n in range(len(self._model_external))]

        shifted_torsions = [\
            coupled[:,self._torsion_inds[n]:self._torsion_inds[n+1]] \
                for n in range(len(self._model_external))]

        return self.merge_dofs(external, bonds, angles, shifted_torsions)

    def logpdf(self, bat=None):
        """Calculate the log probability density

        Parameters
        ----------
        bat : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom

        Returns
        -------
        logpdf : numpy.ndarray
            an array with dimensions (N,), with the log probability density

        """
        if bat is None:
            bat = self._bat

        logpdf = np.zeros(bat[0].shape[0])

        (external, bonds, angles, shifted_torsions) = self.split_dofs(bat)

        for n in range(len(self._model_external)):
            if self._model_external[n]:
                logpdf += self._partial_models[f'translation_{n}'].logpdf(\
                    external[n][:,:3])
                logpdf += self._partial_models[f'rotation_{n}'].logpdf(\
                    external[n][:,3:6])

        for n in range(len(self._model_external)):
            logpdf += self._partial_models[f'bonds_{n}'].logpdf(bonds[n])

        if self._coupled_dofs == 'torsion':
            for n in range(len(self._model_external)):
                logpdf += self._partial_models[f'angles_{n}'].logpdf(angles[n])

        if self._coupled_dofs == 'angle_torsion':
            coupled = np.hstack(angles + shifted_torsions)
        elif self._coupled_dofs == 'torsion':
            coupled = np.hstack(shifted_torsions)
        logpdf += self._partial_models['coupled'].logpdf(coupled)

        return logpdf

    def split_dofs(self, bat):
        """Split a coordinate array into separate arrays

        Parameters
        ----------
        bat : list of numpy.ndarray
            Each element in the list is an array with dimensions (N, 3A),
            where N is the number of samples and A is the number of atoms.
        """
        return split_dofs(bat, self._n_torsions, self._torsion_shifts)

    def merge_dofs(self, external, bonds, angles, shifted_torsions):
        """Merge arrays for separate degrees of freedom into a list of single arrays

        Parameters
        ----------
        external : list of numpy.ndarray
            The external degrees of freedom, translation and rotation.
            An array with dimensions (N, 6), where N is the number of samples.
        bonds : list of numpy.ndarray
            Bond lengths, starting with r01 and r12, the distances between the
            root atoms. An array with dimensions (N, A-1), where A is the number
            of atoms.
        angles : list of numpy.ndarray
            Bond angles, starting with a0123, the angle between the root atoms.
            An array with dimensions (N, A-2), where A is the number of atoms.
        shifted_torsions : list of numpy.ndarray
            Torsion angles, shifted so minimize probability density at the
            periodic boundary. An array with dimensions (N, A-3), where
            A is the number of atoms.

        Returns
        -------
        bat : list of numpy.ndarray
            An array with dimensions (N, 3A), where N is the number of samples
            and A is the number of atoms. If bat is None, then it will be taken
            from the :class:`MDAnalysis.analysis.bat.BAT` instance obtained
            during initialization..
        """
        bat = []

        for n in range(len(external)):
            if external[n] is None:
                external[n] = np.tile(self._reference_external[n], \
                                      (bonds[n].shape[0], 1))
            torsions = shifted_torsions[n] + self._torsion_shifts[n]
            torsions[torsions > np.pi] -= tau
            bat.append(np.hstack([external[n], \
                bonds[n][:,:2], angles[n][:,:1], \
                bonds[n][:,2:], angles[n][:,1:], \
                torsions]))
        return bat

    def _extract_lnZ(self):
        """Extracts the log partition function from partial models

        The partial models should be in the self._partial_models dictionary

        Returns
        -------
        lnZ : dict of float
            The log partition function for each partial model and the total
        """
        lnZ = {}
        for key in self._partial_models.keys():
            lnZ[key] = self._partial_models[key].lnZ
            if hasattr(self._partial_models[key], 'lnZ_J'):
                lnZ[key + '_J'] = getattr(self._partial_models[key], 'lnZ_J')
        lnZ['total'] = np.sum([lnZ[key] for key in lnZ.keys()])
        return lnZ
