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
r"""BAT Ensemble Models --- :mod:`MDAnalysis.analysis.encore.bat_models`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains classes that model ensembles based on their
Bond-Angle-Torsion (BAT) coordinates. Classes in this module can be used
by themselves or with :class:`MDAnalysis.analysis.encore.free_energy.FreeEnergy`.


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
   from MDAnalysis.analysis.encore.bat_models import IndependentGaussianModel

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
from .partial_models.independentkde import IndependentKDE
from .partial_models.principalcomponentskde import PrincipalComponentsKDE
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
    A : np.array
        an array with dimensions (N, K), where N is the number of samples
        and K is the number of degrees of freedom.

    Returns
    -------
    shifts : np.array
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


class EnsembleModelBase:
    """Base class to model an ensemble as a probability distribution

    Subclasses should implement _setup_partial_models.
    If the partial models are not bonds, angles, shifted_torsions,
    translation, and rotation, then rvs and logpdf need to be reimplemented.

    Attributes
    ----------
    lnZ : dict
        Log configurational integral/normalizing constant for different
        partial models
    _partial_models : dict
        Partial models used by subclasses to implement rvs and logpdf

    """
    def __init__(self, bat, model_external=False, **kwargs):
        r"""Parameters
        ----------
        bat : np.array
            an array with dimensions (N,3A), where A is the number of atoms.
            The columns are ordered with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.
        model_external : bool
            Whether to model external degrees of freedom or not
        """
        self._bat = bat
        self._model_external = model_external

        self.n_torsions = int((bat.shape[1] - 9) / 3)
        self._reference_external = np.copy(bat[0][:6])
        self._torsion_shifts = measure_torsion_shifts(\
            bat[:, 2 * self.n_torsions + 9:])

        # Sets up partial models and extracts their partition functions
        self._partial_models = {}
        self._setup_partial_models()
        self.lnZ = self._extract_lnZ()

    def _setup_partial_models(self):
        """Sets up partial models

        This function is the crux of distinctions between subclasses
        """
        raise NotImplementedError

    def rvs(self, N):
        """Generate random samples

        Parameters
        ----------
        N : int
            number of samples to generate

        Returns
        -------
        bat : np.array
            an array with dimensions (N,3A), where A is the number of atoms.
            The columns are ordered with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.

        """
        if self._model_external:
          external = np.hstack([\
              self._partial_models['translation'].rvs(N), \
              self._partial_models['rotation'].rvs(N) \
          ])
        else:
          external = None
        return self.merge_dofs(
          external, \
          self._partial_models['bonds'].rvs(N),
          self._partial_models['angles'].rvs(N),
          self._partial_models['shifted_torsions'].rvs(N)
        )

    def logpdf(self, bat=None):
        """Calculate the log probability density

        Parameters
        ----------
        bat : np.array
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom

        Returns
        -------
        logpdf : np.array
            an array with dimensions (N,), with the log probability density

        """
        if bat is None:
            bat = self._bat

        (external, bonds, angles, shifted_torsions) = self.split_dofs(bat)
        logpdf = \
            self._partial_models['bonds'].logpdf(bonds) + \
            self._partial_models['angles'].logpdf(angles) + \
            self._partial_models['shifted_torsions'].logpdf(shifted_torsions)
        if self._model_external:
          logpdf += self._partial_models['translation'].logpdf(external[:,:3])
          logpdf += self._partial_models['rotation'].logpdf(external[:,3:6])
        return logpdf

    def split_dofs(self, bat=None):
        """Split a coordinate array into separate arrays

        Parameters
        ----------
        bat : np.array
            An array with dimensions (N, 3A), where N is the number of samples
            and A is the number of atoms. If bat is None, then it will be taken
            from the :class:`MDAnalysis.analysis.bat.BAT` instance obtained
            during initialization.

        Returns
        -------
        external : np.array
            The external degrees of freedom, translation and rotation.
            An array with dimensions (N, 6), where N is the number of samples.
        bonds : np.array
            Bond lengths, starting with r01 and r12, the distances between the
            root atoms. An array with dimensions (N, A-1), where A is the number
            of atoms.
        angles : np.array
            Bond angles, starting with a0123, the angle between the root atoms.
            An array with dimensions (N, A-2), where A is the number of atoms.
        shifted_torsions : np.array
            Torsion angles, shifted so minimize probability density at the
            periodic boundary. An array with dimensions (N, A-3), where
            A is the number of atoms.
        """

        if bat is None:
            bat = self._bat

        external = bat[:,:6]
        bond_indices = [6, 7] + list(range(9, self.n_torsions + 9))
        angle_indices = [8] + list(range(self.n_torsions + 9, \
          2 * self.n_torsions + 9))
        bonds = bat[:, bond_indices]
        angles = bat[:, angle_indices]
        shifted_torsions = np.copy(bat[:, 2 * self.n_torsions + 9:])
        shifted_torsions -= self._torsion_shifts
        shifted_torsions[shifted_torsions < -np.pi] += tau

        return (external, bonds, angles, shifted_torsions)

    def merge_dofs(self, external, bonds, angles, shifted_torsions):
        """Merge arrays for separate degrees of freedom into a single array

        Parameters
        ----------
        external : np.array or None
            The external degrees of freedom, translation and rotation.
            If if it is an array, it should have dimensions (N, 6),
            where N is the number of samples.
            If if it is None, then external degrees of freedom from the first
            frame of initial trajectory will be used.
        bonds : np.array
            Bond lengths, starting with r01 and r12, the distances between the
            root atoms. An array with dimensions (N, A-1), where A is the number
            of atoms.
        angles : np.array
            Bond angles, starting with a0123, the angle between the root atoms.
            An array with dimensions (N, A-2), where A is the number of atoms.
        shifted_torsions : np.array
            Torsion angles, shifted so minimize probability density at the
            periodic boundary. An array with dimensions (N, A-3), where
            A is the number of atoms.

        Returns
        -------
        bat : np.array
            An array with dimensions (N, 3A), where N is the number of samples
            and A is the number of atoms. If bat is None, then it will be taken
            from the :class:`MDAnalysis.analysis.bat.BAT` instance obtained
            during initialization..
        """
        if external is None:
            external = np.tile(self._reference_external, (bonds.shape[0], 1))
        torsions = shifted_torsions + self._torsion_shifts
        torsions[torsions > np.pi] -= tau
        return np.hstack([external, \
            bonds[:,:2], angles[:,:1], \
            bonds[:,2:], angles[:,1:], \
            torsions])

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

class IndependentBATEnsembleModel(EnsembleModelBase):
    """Ensemble model in which BAT coordinates are independent from each other

    Distributions of bond lengths and angles will be treated
    as independent Gaussians with parameters from sample estimates.
    The distribution of torsion angles will be treated by different
    partial models depending on the torsion_model parameter.
    These partial models can incorporate correlations between
    different torsion angles.
    External degrees of freedom are based on kernel density estimates.

    """
    def __init__(self, bat,
            torsion_model='IndependentGaussian', \
            source_model=None, \
            **kwargs):
        """{0}
        torsion_model : str
            The model for shifted torsion coordinates
        source_model : IndependentBATEnsembleModel
            Reuses partial models from the source model
        """.format(super(IndependentBATEnsembleModel, self).__init__.__doc__)
        self._torsion_model = torsion_model
        self._source_model = source_model
        super(IndependentBATEnsembleModel, self).__init__(bat, **kwargs)


    def _setup_partial_models(self):
        """Sets up partial models

        Independent Gaussian models are used for the bond lengths and angles.
        A multivariate Guassian is used for torsions. External degrees of
        freedom are based on kernel density estimates.

        """
        (external, bonds, angles, shifted_torsions) = self.split_dofs()

        if self._source_model is None:
            # Define the partial models based on the sampled data
            if self._model_external:
                self._partial_models['translation'] = \
                    Translational(external[:,:3])
                self._partial_models['rotation'] = \
                    Rotational(external[:,3:6])
            self._partial_models['bonds'] = \
                IndependentGaussian(bonds, 'bond')
            self._partial_models['angles'] = \
                IndependentGaussian(angles, 'angle')
        else:
            # Reuse partial models from source model
            for subset in ['translation', 'rotation', 'bonds', 'angles']:
                if subset in self._source_model._partial_models.keys():
                    self._partial_models[subset] = \
                        self._source_model._partial_models[subset]

        if self._torsion_model=='IndependentGaussian':
            self._partial_models['shifted_torsions'] = \
                IndependentGaussian(shifted_torsions, 'torsion')
        elif self._torsion_model=='IndependentKDE':
            self._partial_models['shifted_torsions'] = \
                IndependentKDE(shifted_torsions, 'torsion')
        elif self._torsion_model=='MultivariateGaussian':
            self._partial_models['shifted_torsions'] = \
                MultivariateGaussian(shifted_torsions, 'torsion')
        elif self._torsion_model=='PrincipalComponentsKDE':
            self._partial_models['shifted_torsions'] = \
                PrincipalComponentsKDE(shifted_torsions, 'torsion')
