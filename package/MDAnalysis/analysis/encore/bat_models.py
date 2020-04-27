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
Bond-Angle-Torsion (BAT) coordinates.

TODO: Write intro to module. This module contains ... Citation example: [Minh2020]_.


See Also
--------
:func:`~MDAnalysis.analysis.encore.similarity.hes()`
    function that compares two ensembles after representing each as a harmonic
    oscillator


Example applications
--------------------

TODO: Write this. The :class:`~MDAnalysis.analysis.bat.BAT` class defines bond-angle-torsion
coordinates based on the topology of an atom group and interconverts between
Cartesian and BAT coordinate systems. For example, we can determine internal
coordinates for residues 5-10 of adenylate kinase (AdK). The trajectory is
included within the test data files::

   import MDAnalysis as mda
   from MDAnalysisTests.datafiles import PSF, DCD
   import numpy as np

   u = mda.Universe(PSF, DCD)

   # selection of atomgroups
   selected_residues = u.select_atoms("resid 5-10")

   from MDAnalysis.analysis.bat import BAT
   R = BAT(selected_residues)

   # Calculate BAT coordinates for a trajectory
   R.run()

   # Reconstruct Cartesian coordinates from BAT coordinates
   bat = R.bat[0]
   XYZ = R.Cartesian(bat)

   # The difference between the original and reconstructed coordinates
   # should be zero.
   print(np.sum(np.abs(XYZ - selected_residues.positions)>1E-6))

After R.run(), the coordinates can be accessed with :attr:`R.bat`.


References
----------

.. [Minh2020] Minh, David D L (2020). "Alchemical Grid Dock (AlGDock): Binding
   Free Energy Calculations between Flexible Ligands and Rigid Receptors."
   *Journal of Computational Chemistry* 41(7): 715–30.
   doi:`10.1002/jcc.26036 <https://doi.org/10.1002/jcc.26036>`_

TODO: Format this
    Adapted measure_torsion_shifts and fitting from

    Gyimesi G, Zavodszky P, Szilagyi A:
    Calculation of configurational entropy differences from conformational ensembles using
    Gaussian mixtures.
    J. Chem. Theory Comput., (Just Accepted Manuscript)
    DOI: 10.1021/acs.jctc.6b00837
    http://gmentropy.szialab.org/

"""
from __future__ import absolute_import
# from six.moves import zip, range

import numpy as np

import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.bat import BAT

from MDAnalysis.analysis.encore.partial_models.gaussian import Gaussian
from MDAnalysis.analysis.encore.partial_models.external import Translational, Rotational

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


class EnsembleModelBase(AnalysisBase):
    """Base class to model an ensemble as a probability distribution

    Model based on this class should implement the methods rvs and logpdf.

    """
    def __init__(self, BAT, model_external=False, **kwargs):
        r"""Parameters
        ----------
        BAT : MDAnalysis.analysis.bat.BAT
            an instance of :class:`MDAnalysis.analysis.bat.BAT` for the
            AtomGroup of interest
        model_external : bool
            Whether to model external degrees of freedom or not
        """
        super(EnsembleModelBase, self).__init__(\
            BAT.getAtomGroup().universe.trajectory, **kwargs)
        self._BAT = BAT
        self._model_external = model_external

        # If BAT coordinates are not available, run the calculation
        if (not hasattr(self._BAT, 'bat')) or (self._BAT.bat == []):
            self._BAT.run()

        bat = np.array(self._BAT.bat)
        self.n_torsions = int((bat.shape[1] - 9) / 3)
        self._reference_external = np.copy(bat[0][:6])
        self._torsion_shifts = measure_torsion_shifts(\
            bat[:, 2 * self.n_torsions + 9:])

        self._partial_models = {}

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
        raise NotImplementedError

    def logpdf(self, bat):
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
        raise NotImplementedError

    def split_dofs(self, bat=None):
        """Split a coordinate array into separate arrays

        Parameters
        ----------
        bat : np.array
            An array with dimensions (N, 3A), where N is the number of samples
            and A is the number of atoms. If bat is None, then it will be taken
            from the :class:`MDAnalysis.analysis.bat.BAT` instance used to
            initialize the class.

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
            bat = np.copy(self._BAT.bat)

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
            from the :class:`MDAnalysis.analysis.bat.BAT` instance used to
            initialize the class.
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


class GaussianModel(EnsembleModelBase):
    """Models an ensemble as independent Gaussians

    Distributions of bond lengths, angles, and torsions will be treated
    as independent Gaussians with parameters from sample estimates.

    """
    def __init__(self, BAT, model_external=False, **kwargs):
        r"""Parameters
        ----------
        BAT : MDAnalysis.analysis.bat.BAT
            an instance of :class:`MDAnalysis.analysis.bat.BAT` for the
            AtomGroup of interest

        """
        super(GaussianModel, self).__init__(BAT, model_external, **kwargs)

        (external, bonds, angles, shifted_torsions) = self.split_dofs()

        # Define the partial models based on the sampled data
        if self._model_external:
          self._partial_models['translation'] = Translational(external[:,:3])
          self._partial_models['rotation'] = Rotational(external[:,3:6])
        self._partial_models['bonds'] = Gaussian(bonds, 'bond')
        self._partial_models['angles'] = Gaussian(angles, 'angle')
        self._partial_models['shifted_torsions'] = \
            Gaussian(shifted_torsions, 'torsion')

        # Extract the partition function from partial models
        self.lnZ = self._extract_lnZ()
        # TODO: Free energy of transferring into Gaussian model

    def rvs(self, N):
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
        if bat is None:
            bat = np.copy(self._BAT.bat)

        (external, bonds, angles, shifted_torsions) = self.split_dofs(bat)
        logpdf = \
            self._partial_models['bonds'].logpdf(bonds) + \
            self._partial_models['angles'].logpdf(angles) + \
            self._partial_models['shifted_torsions'].logpdf(shifted_torsions)
        if self._model_external:
          logpdf += self._partial_models['translation'].logpdf(external[:,:3])
          logpdf += self._partial_models['rotation'].logpdf(external[:,3:6])
        return logpdf


# Other possible models include KDEModel, DimredModel, and GaussianMixtureModel
