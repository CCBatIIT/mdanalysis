# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v21 or any higher version
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
r"""Independent Gaussian partial model --- :mod:`MDAnalysis.analysis.tem.partial_models.independentgaussian`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling a subset of the degrees of
freedom as independent Gaussian distributions.

"""

import numpy as np
tau = 2 * np.pi

from .base import PartialModelBase

class IndependentGaussian(PartialModelBase):
    """Models a subset of the degrees of freedom as independent Gaussians

    The mean and standard deviation is based on the sample mean and standard
    deviation.

    If the sample standard deviation is below a cutoff value, std_cutoff,
    the degree of freedom is treated as constrained to the mean. It does
    not contribute to the partition function or the log probability.

    The Jacobian is dependent on whether the degree of freedom is
    a bond, angle, or torsion. It is approximated as a constant, based on
    the mean value for the degree of freedom.

    """

    _param_keys = ['coordinate_type', 'means', 'stdevs', 'std_cutoff']
    _allowed_coordinate_types = ['bond', 'angle', 'torsion', 'angle_torsion']

    def __init__(self, coordinate_type, means, stdevs, std_cutoff=0.01):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        means : numpy.ndarray
            the mean value of each coordinate,
            an array with dimensions (K,)
        stdevs : numpy.ndarray
            the standard deviation of each coordinate,
            an array with dimensions (K,)
        std_cutoff : float
            the minimum standard deviation for a coordinate
            to be considered a degree of freedom in the model
        """
        super(IndependentGaussian, self).__init__(coordinate_type)

        # The degrees of freedom have a standard deviation above the cutoff
        is_dof = (stdevs > std_cutoff)
        K = np.sum(is_dof)

        # Calculate the log normalizing constant, including the Jacobian factor
        if coordinate_type == 'bond':
            # For the bond length, $b$, the Jacobian is $b^2$.
            lnZ_J = 2 * np.sum(np.log(means))
        elif coordinate_type == 'angle':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            lnZ_J = 2 * np.sum(np.log(np.sin(means)))
        elif coordinate_type == 'angle_torsion':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            # The first half of the coordinates are angles and the rest torsions
            lnZ_J = 2 * np.sum(np.log(np.sin(means[:len(means)/2])))
        elif coordinate_type == 'torsion':
            # For torsions, the Jacobian is unity
            lnZ_J = 0.
        # For a Gaussian distribution,
        # $Z = \sqrt{2 \pi} \sigma$
        # $\ln(Z) = 1/2 \ln (2 \pi} + \ln (\sigma)$
        lnZ = np.log(tau) * K / 2. + \
            np.sum(np.log(stdevs[is_dof]))

        self._means = means
        self._stdevs = stdevs
        self._std_cutoff = std_cutoff
        self._is_dof = is_dof
        self.K = K
        self.lnZ = lnZ
        self.lnZ_J = lnZ_J

    @classmethod
    def from_data(cls, coordinate_type, X, std_cutoff=0.01):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        std_cutoff : float
            the minimum standard deviation for a coordinate
            to be considered a degree of freedom in the model
        """
        # Determine parameters
        means = np.mean(X, 0)
        stdevs = np.std(X, 0)
        return cls(coordinate_type, means, stdevs, std_cutoff)

    def rvs(self, N):
        X = np.zeros((N,self._means.shape[0]))
        stdrandn = np.random.randn(N, self.K)
        rvs = self._stdevs[self._is_dof] * stdrandn + \
            self._means[self._is_dof]
        X[:,self._is_dof] = rvs
        X[:,~self._is_dof] = self._means[~self._is_dof]
        return X

    def logpdf(self, X):
        stddelta = (X[:,self._is_dof] - self._means[self._is_dof]) / \
                    self._stdevs[self._is_dof]
        logpdf = -np.sum(np.square(stddelta) / 2., axis=1) - self.lnZ
        return logpdf
