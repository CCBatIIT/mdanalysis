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
r"""Independent kernel density estimate model --- :mod:`MDAnalysis.analysis.tem.partial_models.independentkde`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling a subset of the degrees of
freedom as independent Gaussian kernel density estimates.

"""

import numpy as np
tau = 2 * np.pi

from scipy.stats import gaussian_kde
from scipy.stats import rv_discrete

from .base import PartialModelBase
from MDAnalysis.analysis.encore.covariance import shrinkage_covariance_estimator

class IndependentKDE(PartialModelBase):
    """Models a subset of the degrees of freedom as independent KDEs

    By default, each degree of freedom is discretized. Continuous versions
    can be accessed by running rvs and logpdf with discretize=False.

    The mean and standard deviation is based on the sample mean and standard
    deviation. The Jacobian is dependent on whether the degree of freedom is
    a bond, angle, or torsion. It is approximated as a constant, based on
    the mean value for the degree of freedom.

    """

    _param_keys = ['coordinate_type', 'X', 'nbins']
    _allowed_coordinate_types = ['bond', 'angle', 'torsion', 'angle_torsion']

    def __init__(self, coordinate_type, X, nbins=250):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        nbins : int
            The number of bins to use in discretizing each degree of freedom
        """
        super(IndependentKDE, self).__init__(coordinate_type)

        # Initiate kernel density estimate instances
        self.K = X.shape[1]
        self._kde = []
        for dim in range(self.K):
            self._kde.append(gaussian_kde(X[:, dim]))

        # Variables for discretized KDE
        self._edges = []
        self._delta = []
        self._centers = []
        self._rv_discrete = []
        self._ln_norm = 0.
        for dim in range(self.K):
            x_min = min(X[:, dim])
            x_max = max(X[:, dim])
            tail = (x_max - x_min) * 0.1
            edges = np.linspace(x_min - tail, x_max + tail, nbins)
            delta = edges[1] - edges[0]
            centers = edges[:-1] + delta / 2
            rho = self._kde[dim](centers)
            norm = np.sum(rho)
            rho /= norm
            rv = rv_discrete(name=repr(dim), values=(range(len(centers)), rho))
            self._edges.append(edges)
            self._delta.append(delta)
            self._centers.append(centers)
            self._rv_discrete.append(rv)
            self._ln_norm += np.log(norm)

        # Calculate the log normalizing constant, including the Jacobian factor
        if coordinate_type == 'bond':
            # For the bond length, $b$, the Jacobian is $b^2$.
            lnZ_J = 2 * np.sum(np.log(np.mean(X, 0)))
        elif coordinate_type == 'angle':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            lnZ_J = 2 * np.sum(np.log(np.sin(np.mean(X, 0))))
        elif coordinate_type == 'angle_torsion':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            # The first half of the coordinates are angles and the rest torsions
            lnZ_J = 2 * np.sum(np.log(np.sin(means[:X.shape[1]/2])))
        elif (coordinate_type == 'torsion') or \
             (coordinate_type == 'translation'):
            # For torsions, the Jacobian is unity
            lnZ_J = 0.
        # The log normalizing constant is zero because the kde is normalized
        lnZ = 0.

        self._X = X
        self._nbins = nbins
        self.lnZ = lnZ
        self.lnZ_J = lnZ_J

    def rvs(self, N, discretize=True):
        if discretize:
            samples = [self._centers[dim][self._rv_discrete[dim].rvs(size=N)] \
                for dim in range(self.K)]
            return numpy.ndarray(samples).transpose()
        else:
            return np.hstack([kde.resample(N).transpose() \
                for kde in self._kde])

    def logpdf(self, X, discretize=True):
        if discretize:
            return np.sum([self._rv_discrete[dim].logpmf(\
                np.floor((X[:,dim] - self._edges[dim][0])/self._delta[dim])) \
                    for dim in range(self.K)],0) + self._ln_norm
        else:
            return np.sum([self._kde[dim].logpdf(X[:,dim]) \
                for dim in range(self.K)], axis = 0)

class PrincipalComponentsKDE(IndependentKDE):
    """Models a subset of the degrees of freedom with kernel density estimates on PCA

    """

    _param_keys = ['coordinate_type', 'X_pca', \
        'means', 'eigenvalues', 'eigenvectors', 'nbins']
    _allowed_coordinate_types = ['bond', 'angle', 'torsion', 'angle_torsion', \
        'translation']

    def __init__(self, coordinate_type, X_pca, \
                means, eigenvalues, eigenvectors, nbins=1000):
        # Use self.X_pca to initiate IndependentKDE instance
        super(PrincipalComponentsKDE, self).__init__(\
            coordinate_type, X_pca, nbins)

        # Correct log normalizing constants with original mean values
        if coordinate_type == 'bond':
            # For the bond length, $b$, the Jacobian is $b^2$.
            lnZ_J = 2 * np.sum(np.log(means))
        elif coordinate_type == 'angle':
            # For the bond angle, $\theta$, the Jacobian is $sin(\theta)$
            lnZ_J = 2 * np.sum(np.log(np.sin(means)))
        elif (coordinate_type == 'torsion') or \
             (coordinate_type == 'translation'):
            # For torsions, the Jacobian is unity
            lnZ_J = 0.
        # The log normalizing constant is zero because the kde is normalized
        lnZ = 0.

        if (coordinate_type == 'translation'):
            # Log volume of the binding site
            box = [(self._edges[dim][-1] - self._edges[dim][0]) \
                for dim in range(3)]
            self.lnV_site = np.sum(np.log(box))
            # Standard state correction for confining the system into a box
            # The standard state volume for a single molecule
            # in a box of size 1 L is 1.66053928 nanometers**3
            self.DeltaG_xi = -self.lnV_site + np.log(1660.53928)

        self._X_pca = X_pca
        self._means = means
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._nbins = nbins
        self.lnZ = lnZ
        self.lnZ_J = lnZ_J

    @classmethod
    def from_data(cls, coordinate_type, X, nbins=1000):
        # Performs principal components analysis
        means = np.mean(X, 0)
        X_c = X - means
        cov = shrinkage_covariance_estimator(X_c)
        [eigenvalues, eigenvectors] = np.linalg.eig(cov)
        X_pca = np.dot(X_c, eigenvectors)
        return cls(coordinate_type, X_pca, \
            means, eigenvalues, eigenvectors, nbins)

    def rvs(self, N, discretize=True):
        X_pca = super(PrincipalComponentsKDE, self).rvs(N, discretize)
        return np.dot(X_pca, self._eigenvectors) + self._means

    def logpdf(self, X, discretize=True):
        X_c = X - self._means
        X_pca = np.dot(X_c, self._eigenvectors)
        return super(PrincipalComponentsKDE, self).logpdf(X_pca, discretize)
