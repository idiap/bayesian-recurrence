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
Here is where the simple Bayesian Recurrent Unit (sBRU)
is defined as a PyTorch module.
"""
import torch
import torch.nn as nn

# ----------------
# -- Simple BRU --
# ----------------


class sBRU(nn.Module):
    """
    This function implements a simple BRU (sBRU)

    sBRU is a Bayesian unit with layer-wise recurrence and a
    feedback on log-probabilities. It represents a probabilistic
    version of a standard RNN without gates.

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
        Type of hidden state, either 'probs', 'logprobs' or 'rnn'. The 'probs'
        and 'logprobs' are two equivalent formulations of our unit, where
        hidden states represents probabilities and log-probabilities
        respectively. The 'rnn' formulation is also given here for comparison
        with standard networks.
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
        super(sBRU, self).__init__()

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
        self.eps = 1e-11

        if hidden_type not in ["probs", "logprobs", "rnn"]:
            raise ValueError(f"Invalid hidden type {hidden_type}")

        # Debug mode
        torch.autograd.set_detect_anomaly(False)

        # Initialize lists
        self.W = nn.ModuleList([])
        self.V = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # Loop over hidden layers
        for lay in range(self.nb_layers):
            nb_hiddens = layer_sizes[lay]

            # Initialize trainable parameters
            self.W.append(nn.Linear(nb_inputs, nb_hiddens, bias=use_bias))
            self.V.append(nn.Linear(nb_hiddens, nb_hiddens, bias=False))
            nn.init.xavier_uniform_(self.W[lay].weight)
            nn.init.orthogonal_(self.V[lay].weight)

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
                _norm = self.norm[lay](
                    Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2])
                )
                Wx = _norm.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

            # Process time steps
            if self.hidden_type == "probs":
                h = self.sbru_cell_probs(Wx, self.V[lay])
            elif self.hidden_type == "logprobs":
                h = self.sbru_cell_logprobs(Wx, self.V[lay])
            elif self.hidden_type == "rnn":
                h = self.rnn_cell(Wx, self.V[lay])

            # Concatenate forward and backward sequences on feat dim
            if self.bidirectional:
                h_f, h_b = h.chunk(2, dim=0)
                h_b = h_b.flip(1)
                h = torch.cat([h_f, h_b], dim=2)

            # Update for next layer
            x = h

        return x

    def sbru_cell_probs(self, Wx, V):
        """
        Returns the hidden states for all time steps
        as probabilities (see equation 13 in the paper).

        Arguments
        ---------
        Wx : torch.Tensor
            Linearly transformed input
        V : torch.Tensor
            Recurrent weight matrix
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            ht = torch.sigmoid(Wx[:, t, :] + V(torch.log(ht + self.eps)))
            h.append(ht)

        return torch.stack(h, dim=1)

    def sbru_cell_logprobs(self, Wx, V):
        """
        Returns the hidden states for all time steps as
        log-probabilities (see equation 14 in paper).
        Note that using a relu instead of the softplus also works
        as the former can be seen as an approximation of the latter.

        Arguments
        ---------
        Wx : torch.Tensor
            Linearly transformed input
        V : torch.Tensor
            Recurrent weight matrix
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            ht = self.softplus(Wx[:, t, :] + V(ht))
            h.append(ht)

        return torch.stack(h, dim=1)

    def rnn_cell(self, Wx, V):
        """
        Returns the hidden states for all time steps
        with the standard RNN formulation. Here the feedback
        is not on log-probabilities (as it should be from our
        derivation), but on probabilities.

        Arguments
        ---------
        Wx : torch.Tensor
            Linearly transformed input
        V : torch.Tensor
            Recurrent weight matrix
        """
        # Initialize hidden state
        h = []
        ht = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)

        # Loop over time
        for t in range(Wx.shape[1]):
            ht = torch.sigmoid(Wx[:, t, :] + V(ht))
            h.append(ht)

        return torch.stack(h, dim=1)
