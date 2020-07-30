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
r"""Multivariate Gaussian partial model --- :mod:`MDAnalysis.analysis.tem.partial_models.multivariategaussian`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling a subset of the degrees of
freedom as a multivariate Gaussian distributions.

"""

import numpy as np
tau = 2 * np.pi

from .base import PartialModelBase
from MDAnalysis.analysis.encore.covariance import shrinkage_covariance_estimator
from scipy.stats import multivariate_normal

class MultivariateGaussian(PartialModelBase):
    """Models a subset of the degrees of freedom as a multivariate Gaussian

    The mean and standard deviation is based on the sample mean and covariance
    matrix a shrinkage estimator. The Jacobian is dependent on whether
    the degree of freedom is a bond, angle, or torsion. It is approximated
    as a constant, based on the mean value for the degree of freedom.

    """

    _param_keys = ['coordinate_type', 'means', 'cov']
    _allowed_coordinate_types = ['bond', 'angle', 'torsion', 'angle_torsion']

    def __init__(self, coordinate_type, means, cov):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        means : numpy.ndarray
            the mean value of each coordinate, an array with dimensions (K,)
        cov : numpy.ndarray
            the covariance matrix, an array with dimensions (K, K)
        """
        super(MultivariateGaussian, self).__init__(coordinate_type)

        # Determine parameters
        mvn = multivariate_normal(means, cov)
        K = len(means)

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

        # Logpdf results are normalized
        lnZ = 0.
        # For a multivariate Gaussian distribution,
        # Z = (2 \pi)^(K/2) |\Sigma|^{1/2}
        # ln Z = K/2 \ln (2 \pi) + 1/2 \ln (|\Sigma|)
        # (signdet, lndet) = np.linalg.slogdet(cov)
        # if signdet == 1.0:
        #     self.lnZ = np.log(tau) * self.K / 2. + lndet / 2.
        # else:
        #     raise ValueError('The sign on the determinant of the covariance' + \
        #                      'matrix is not one.')

        self._means = means
        self._cov = cov
        self._mvn = mvn
        self.K = K
        self.lnZ = lnZ
        self.lnZ_J = lnZ_J

    @classmethod
    def from_data(cls, coordinate_type, X):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        """
        # Determine parameters
        means = np.mean(X, 0)
        cov = shrinkage_covariance_estimator(X)
        return cls(coordinate_type, means, cov)

    def rvs(self, N):
        return self._mvn.rvs(N)

    def logpdf(self, X):
        return self._mvn.logpdf(X)
