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
r"""Gaussian mixture partial model --- :mod:`MDAnalysis.analysis.tem.partial_models.gaussianmixture`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling a subset of the degrees of
freedom as a Gaussian mixture.

"""

import numpy as np
tau = 2 * np.pi

from .base import PartialModelBase
from sklearn import mixture

class GaussianMixture(PartialModelBase):
    """Models a subset of the degrees of freedom as a Gaussian mixture

    Uses expectation-maximization (EM) to fit a Gaussian mixture to the data.
    The number of components is determined by increasing the number of
    components until the score (average log probability of test data)
    decreases, indicating overfitting.

    The Jacobian is dependent on whether the degree of freedom is a bond,
    angle, or torsion. It is approximated as a constant, based on the
    mean value for the degree of freedom.

    """

    _param_keys = ['coordinate_type', 'means', 'gmm']
    _allowed_coordinate_types = ['bond', 'angle', 'torsion', 'angle_torsion']

    def __init__(self, coordinate_type, means, gmm):
        """Parameters
        ----------
        coordinate_type : str
            the type of coordinate
        means : numpy.ndarray
            the mean value of each coordinate,
            an array with dimensions (K,)
        gmm : sklearn.mixture.GaussianMixture
            represents desired distribution for the coordinates
        """
        super(GaussianMixture, self).__init__(coordinate_type)

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
            lnZ_J = 2 * np.sum(np.log(np.sin(means[:K/2])))
        elif coordinate_type == 'torsion':
            # For torsions, the Jacobian is unity
            lnZ_J = 0.

        self._means = means
        self._gmm = gmm
        # In scikit-learn, the log probabilities
        # of the Gaussian mixture are normalized
        # https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/mixture/_gaussian_mixture.py#L380
        self.lnZ = 0.
        self.lnZ_J = lnZ_J

    @classmethod
    def from_data(cls, coordinate_type, X, fraction_train=0.8):
        """Parameters
        ----------
        coordinate_type : str
            "bond" or "angle" or "torsion"
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        fraction_train : float
            fraction of the samples used to train versus test the mixture model
        """
        means = np.mean(X, 0)

        # Split data into training and testing sets
        n_train = int(X.shape[0]*fraction_train)
        inds = np.random.permutation(X.shape[0])
        X_train = X[inds[:n_train],:]
        X_test = X[inds[n_train:],:]

        # Increase the number of components until adding more
        # does not improve the score of the test set
        n_components = 1
        scores = [-np.inf]
        while True:
          gmm_n = mixture.GaussianMixture(n_components=n_components, \
                                          covariance_type='full')
          gmm_n.fit(X_train)
          scores.append(gmm_n.score(X_test))
          if scores[-1]<scores[-2]:
            scores = scores[1:-1]
            n_components -= 1
            break
          gmm = gmm_n
          n_components += 1

        return cls(coordinate_type, means, gmm)

    @classmethod
    def from_dict(cls, param_dict):
        """Parameters
        ----------
        param_dict : dict
            a dictionary of parameters needed to reinitialize the model
        """
        gmm = mixture.GaussianMixture(**param_dict['params'])
        for key in param_dict['fit'].keys():
            setattr(gmm, key, param_dict['fit'][key])
        return cls(param_dict['coordinate_type'], param_dict['means'], gmm)

    def to_dict(self):
        """Return a dictionary of parameters needed to reinitialize the model

        Returns
        -------
        param_dict : dict
            a dictionary of parameters needed to initialize a partial model.
        """
        gmm_params = self._gmm.get_params()
        gmm_fit = dict([(v, getattr(self._gmm, v)) for v in vars(self._gmm) \
            if v.endswith("_") and not v.startswith("__")])
        return {'class':repr(getattr(self, '__class__')), \
            'coordinate_type':self.coordinate_type, \
            'means':self._means, \
            'params':gmm_params, 'fit':gmm_fit}

    def rvs(self, N):
        return self._gmm.sample(N)[0]

    def logpdf(self, X):
        return self._gmm.score_samples(X)