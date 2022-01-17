#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the bayesian-recurrence package
#
"""
Here is where the unit-wise Bayesian Recurrent Unit (uBRU)
is defined as a PyTorch module.
"""
import torch
import torch.nn as nn

# -------------------
# -- Unit-wise BRU --
# -------------------


class uBRU(nn.Module):
    """
    This function implements a unit-wise BRU (uBRU)

    uBRU is a Bayesian unit with unit-wise recurrence based on
    trainable transition probabilities, and an optional
    backward recursion similar to a Kalman smoother.

    A. Bittar and P. Garner, 2022.
    Bayesian Recurrent Units and the Forward-Backward algorithm

    Arguments
    ---------
    nb_inputs : int
        Number of input neurons/features.
    layer_sizes : int list
        List of number of neurons in all hidden layers.
    kalman_backward : bool
        If True, backward recursion through sequence is performed.
    bidirectional : bool
        If True, a bidirectional that scans the sequence in both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    dropout : float
        Dropout rate, must be in [0, 1].
    """

    def __init__(
        self,
        nb_inputs,
        layer_sizes,
        kalman_backward=True,
        bidirectional=False,
        normalization="batchnorm",
        use_bias=False,
        dropout=0.0,
    ):
        super(uBRU, self).__init__()

        # Fixed parameters
        self.nb_inputs = nb_inputs
        self.layer_sizes = layer_sizes
        self.nb_layers = len(layer_sizes)
        self.nb_outputs = layer_sizes[-1]
        self.kalman_backward = kalman_backward
        self.bidirectional = bidirectional
        self.normalization = normalization
        self.use_bias = use_bias
        self.dropout = dropout

        # Debug mode
        torch.autograd.set_detect_anomaly(False)

        # Initialize lists
        self.W = nn.ModuleList([])
        self.probs = nn.ParameterList([])
        self.norm = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # Loop over hidden layers
        for lay in range(self.nb_layers):
            nb_hiddens = layer_sizes[lay]

            # Initialize trainable parameters
            self.W.append(nn.Linear(nb_inputs, nb_hiddens, bias=use_bias))
            self.probs.append(nn.Parameter(torch.zeros(3, nb_hiddens)))
            nn.init.xavier_uniform_(self.W[lay].weight)

            # Initialize normalization
            self.normalize = False
            if normalization == "batchnorm":
                self.norm.append(nn.BatchNorm1d(nb_hiddens, momentum=0.05))
                self.normalize = True
            elif normalization == "layernorm":
                self.norm.append(torch.nn.LayerNorm(nb_hiddens))
                self.normalize = True

            # Initialize dropout
            self.drop.append(nn.Dropout(dropout))

            nb_inputs = nb_hiddens * (1 + bidirectional)

    def forward(self, x):

        # Loop over the layers
        for lay in range(self.nb_layers):

            # Concatenate flipped sequence on batch dim
            if self.bidirectional:
                x_flip = x.flip(1)
                x = torch.cat([x, x_flip], dim=0)

            # Apply dropout
            x = self.drop[lay](x)

            # Compute feedforward outside loop (faster)
            Wx = self.W[lay](x)

            # Apply normalization
            if self.normalize:
                _Wx = self.norm[lay](Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
                Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

            # Compute quantities outside loop (faster)
            r = torch.exp(torch.clamp(Wx, max=10))
            probs = torch.sigmoid(self.probs[lay])
            p1 = probs[1, :]
            p2 = probs[2, :]
            at = probs[0, :]

            # Loop over time axis (forward in time)
            a = []
            p = []
            for t in range(Wx.shape[1]):
                pt = p1 * at + p2 * (1 - at)
                at = pt / (pt + r[:, t] * (1 - pt) + 1e-11)
                p.append(pt)
                a.append(at)

            # Optional backward loop in time (Kalman smoother)
            if self.kalman_backward:
                ht = a[-1]
                h = [ht]
                for t in range(Wx.shape[1] - 2, -1, -1):
                    ht = a[t] * (
                        p1 * ht / p[t + 1] + (1 - p1) * (1 - ht) / (1 - p[t + 1])
                    )
                    h.insert(0, ht)
                h = torch.stack(h, dim=1)
            else:
                h = torch.stack(a, dim=1)

            # Concatenate forward and backward sequences on feat dim
            if self.bidirectional:
                h_f, h_b = h.chunk(2, dim=0)
                h_b = h_b.flip(1)
                h = torch.cat([h_f, h_b], dim=2)

            # Update for next layer
            x = h

        return x
