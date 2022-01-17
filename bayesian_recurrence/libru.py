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
Here is where the light Bayesian Recurrent Unit (liBRU)
is defined as a PyTorch module.
"""
import torch
import torch.nn as nn

# ---------------
# -- Light BRU --
# ---------------


class liBRU(nn.Module):
    """
    This function implements a light BRU (liBRU)

    liBRU is a Bayesian unit with layer-wise recurrence, a
    single gate and a Softplus activation function, so that the
    feedback is on log-probabilities. It is a probabilistic version
    of a light gated recurrent unit (liGRU).

    A. Bittar and P. Garner, 2021.
    A Bayesian Interpretation of the Light Gated Recurrent Unit

    Arguments
    ---------
    nb_inputs : int
        Number of input neurons/features.
    layer_sizes : int list
        List of number of neurons in all hidden layers.
    bidirectional : bool
        If True, a bidirectional that scans the sequence in both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    hidden_type : str
        Type of hidden state, either 'probs', 'logprobs', 'geomean' or 'ligru'.
        The 'probs' and 'logprobs' are two equivalent formulations of our unit,
        where hidden states represents probabilities and log-probabilities
        respectively. The 'geomean' is a first approx of the liBRU with a
        geometric mean of probabilities instead of an arithmetic one when
        applying the gate. By further approximating the softplus as a relu,
        one gets the 'ligru' formulation.
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
        bidirectional=False,
        hidden_type="logprobs",
        normalization="batchnorm",
        use_bias=False,
        dropout=0.0,
    ):
        super(liBRU, self).__init__()

        # Fixed parameters
        self.nb_inputs = nb_inputs
        self.layer_sizes = layer_sizes
        self.nb_layers = len(layer_sizes)
        self.nb_outputs = layer_sizes[-1]
        self.bidirectional = bidirectional
        self.hidden_type = hidden_type
        self.normalization = normalization
        self.use_bias = use_bias
        self.dropout = dropout
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        self.eps = 1e-11

        if hidden_type not in ["probs", "logprobs", "geomean", "ligru"]:
            raise ValueError(f"Invalid hidden type {hidden_type}")

        # Debug mode
        torch.autograd.set_detect_anomaly(False)

        # Initialize lists
        self.W = nn.ModuleList([])
        self.Wz = nn.ModuleList([])
        self.V = nn.ModuleList([])
        self.Vz = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # Loop over hidden layers
        for lay in range(self.nb_layers):
            nb_hiddens = layer_sizes[lay]

            # Initialize trainable parameters
            self.W.append(nn.Linear(nb_inputs, nb_hiddens, bias=use_bias))
            self.Wz.append(nn.Linear(nb_inputs, nb_hiddens, bias=use_bias))
            self.V.append(nn.Linear(nb_hiddens, nb_hiddens, bias=False))
            self.Vz.append(nn.Linear(nb_hiddens, nb_hiddens, bias=False))
            nn.init.xavier_uniform_(self.W[lay].weight)
            nn.init.xavier_uniform_(self.Wz[lay].weight)
            nn.init.orthogonal_(self.V[lay].weight)
            nn.init.orthogonal_(self.Vz[lay].weight)

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
            Wzx = self.Wz[lay](x)

            # Apply normalization
            if self.normalize:
                _norm = self.norm[lay](
                    Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2])
                )
                Wx = _norm.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
                _norm = self.norm[lay](
                    Wzx.reshape(Wzx.shape[0] * Wzx.shape[1], Wzx.shape[2])
                )
                Wzx = _norm.reshape(Wzx.shape[0], Wzx.shape[1], Wzx.shape[2])

            # Process time steps
            if self.hidden_type == "probs":
                h = self.libru_cell_probs(Wx, Wzx, self.V[lay], self.Vz[lay])
            elif self.hidden_type == "logprobs":
                h = self.libru_cell_logprobs(Wx, Wzx, self.V[lay], self.Vz[lay])
            elif self.hidden_type == "geomean":
                h = self.libru_cell_geomean(Wx, Wzx, self.V[lay], self.Vz[lay])
            elif self.hidden_type == "ligru":
                h = self.ligru_cell(Wx, Wzx, self.V[lay], self.Vz[lay])

            # Concatenate forward and backward sequences on feat dim
            if self.bidirectional:
                h_f, h_b = h.chunk(2, dim=0)
                h_b = h_b.flip(1)
                h = torch.cat([h_f, h_b], dim=2)

            # Update for next layer
            x = h

        return x

    def libru_cell_probs(self, Wx, Wzx, V, Vz):
        """
        Returns the hidden states for all time steps
        as probabilities (see equations 19a-19c in paper).

        Arguments
        ---------
        Wx, Wzx : torch.Tensor
            Linearly transformed inputs
        V, Vz : torch.Tensor
            Recurrent weight matrices
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + Vz(torch.log(ht + self.eps)))
            ct = torch.sigmoid(Wx[:, t, :] + V(torch.log(ht + self.eps)))
            ht = zt * ht + (1.0 - zt) * ct
            h.append(ht)

        return torch.stack(h, dim=1)

    def libru_cell_logprobs(self, Wx, Wzx, V, Vz):
        """
        Returns the hidden states for all time steps
        as log-probabilities.

        Arguments
        ---------
        Wx, Wzx : torch.Tensor
            Linearly transformed inputs
        V, Vz : torch.Tensor
            Recurrent weight matrices
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + Vz(ht))
            ct = self.softplus(Wx[:, t, :] + V(ht))
            ht = torch.log(zt * torch.exp(ht) + (1.0 - zt) * torch.exp(ct))
            h.append(ht)

        return torch.stack(h, dim=1)

    def libru_cell_geomean(self, Wx, Wzx, V, Vz):
        """
        Returns the hidden states for all time steps as log-probabilities,
        but using the geometric mean instead of the arithmetic mean when
        applying the gate. This is faster and similar to liGRU, only with
        softplus instead of relu.

        Arguments
        ---------
        Wx, Wzx : torch.Tensor
            Linearly transformed inputs
        V, Vz : torch.Tensor
            Recurrent weight matrices
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + Vz(ht))
            ct = self.softplus(Wx[:, t, :] + V(ht))
            ht = zt * ht + (1.0 - zt) * ct
            h.append(ht)

        return torch.stack(h, dim=1)

    def ligru_cell(self, Wx, Wzx, V, Vz):
        """
        Returns the hidden states for all time steps using the
        liGRU formulation. This can be seen as an approximation
        of the liBRU, with geometric mean instead of arithmetic,
        and with relu activation approximating the softplus.

        Arguments
        ---------
        Wx, Wzx : torch.Tensor
            Linearly transformed inputs
        V, Vz : torch.Tensor
            Recurrent weight matrices
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            zt = torch.sigmoid(Wzx[:, t, :] + Vz(ht))
            ct = self.relu(Wx[:, t, :] + V(ht))
            ht = zt * ht + (1.0 - zt) * ct
            h.append(ht)

        return torch.stack(h, dim=1)
