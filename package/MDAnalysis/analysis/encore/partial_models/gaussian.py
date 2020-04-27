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
r"""Gaussian partial model --- :mod:`MDAnalysis.analysis.encore.partial_models.gaussian`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a class for modelling degrees a subset of the degrees of
freedom as independent Gaussian distributions.

"""

import numpy as np
tau = 2 * np.pi

from MDAnalysis.analysis.encore.partial_models.base import PartialModelBase


class Gaussian(PartialModelBase):
    """Models a subset of the degrees of freedom as independent Gaussians

    The mean and standard deviation is based on the sample mean and standard
    deviation. The Jacobian is dependent on whether the degree of freedom is
    a bond, angle, or torsion. It is approximated as a constant, based on
    the mean value for the degreee of freedom.

    """
    def __init__(self, X, coordinate_type):
        """Parameters
        ----------
        X : np.array
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        coordinate_type : str
            "bond" or "angle" or "torsion"
        """
        if not coordinate_type in ['bond', 'angle', 'torsion']:
            raise ValueError('error: coordinate_type must be ' + \
                             '"bond", "angle", or "torsion"')
        self.coordinate_type = coordinate_type

        # Determine parameters
        self._means = np.mean(X, 0)
        self._stdevs = np.std(X, 0)
        self.K = X.shape[1]

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
        # For a Gaussian distribution,
        # $Z = \sqrt{2 \pi} \sigma$
        # $\ln(Z) = 1/2 \ln (2 \pi} + \ln (\sigma)$
        self.lnZ = np.log(tau) * self.K / 2. + np.sum(np.log(self._stdevs))

    def rvs(self, N):
        stdrandn = np.random.randn(N, self.K)
        return self._stdevs * stdrandn + self._means

    def logpdf(self, X):
        stddelta = (X - self._means) / self._stdevs
        logpdf = -np.sum(np.square(stddelta) / 2., axis=1) - self.lnZ
        return logpdf
