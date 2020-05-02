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
r"""Principal components kernel density estimate model --- :mod:`MDAnalysis.analysis.encore.partial_models.principalcomponentskde`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling a subset of the degrees of
freedom as Gaussian kernel density estimates on their principal components

"""

import numpy as np
tau = 2 * np.pi

from ..covariance import shrinkage_covariance_estimator
from scipy.stats import gaussian_kde

from .base import PartialModelBase


class PrincipalComponentsKDE(PartialModelBase):
    """Models a subset of the degrees of freedom as independent Gaussians

    The mean and standard deviation is based on the sample mean and standard
    deviation. The Jacobian is dependent on whether the degree of freedom is
    a bond, angle, or torsion. It is approximated as a constant, based on
    the mean value for the degree of freedom.

    """
    def __init__(self, X, coordinate_type):
        """Parameters
        ----------
        X : np.array
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        coordinate_type : str
            "bond" or "angle" or "torsion" or None
        """
        if not coordinate_type in ['bond', 'angle', 'torsion', None]:
            raise ValueError('error: coordinate_type must be ' + \
                             '"bond", "angle", "torsion", or None')
        self.coordinate_type = coordinate_type

        # Performs principal components analysis
        X_mean = np.mean(X, 0)
        X_c = X - X_mean
        cov = shrinkage_covariance_estimator(X_c)
        [w, v] = np.linalg.eig(cov)
        X_pca = np.dot(X_c, v)
        self._pca = {'mean': X_mean, 'eigenvalues': w, 'eigenvectors': v}

        # Initiate kernel density estimate instances
        self.K = X.shape[1]
        self._kde = []
        for dim in range(self.K):
            self._kde.append(gaussian_kde(X_pca[:, dim]))

        # Appropriate intervals to integrate over
        self._intervals = []
        for dim in range(3):
            x_min = min(X_pca[:, dim])
            x_max = max(X_pca[:, dim])
            tail = (x_max - x_min) * 0.1
            self._intervals.append((x_min - tail, x_max + tail))

        # Calculate the log normalizing constant, including the Jacobian factor
        if coordinate_type == 'bond':
            # For the bond length, $b$, the Jacobian is $b^2$.
            self.lnZ_J = 2 * np.sum(np.log(self._means))
        elif coordinate_type == 'angle':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            self.lnZ_J = 2 * np.sum(np.log(np.sin(self._means)))
        elif coordinate_type == 'torsion':
            # For torsions, the Jacobian is unity
            self.lnZ_J = 0.
        # The log normalizing constant is zero because the kde is normalized
        self.lnZ = 0.

    def rvs(self, N):
        X_pca = np.hstack([kde.resample(N).transpose() \
            for kde in self._kde])
        return np.dot(X_pca, self._pca['eigenvectors']) + \
            self._pca['mean']

    def logpdf(self, X):
        X_c = X - self._pca['mean']
        X_pca = np.dot(X_c, self._pca['eigenvectors'])
        return np.sum([self._kde[dim].logpdf(X_pca[:,dim]) \
            for dim in range(3)], axis = 0)
