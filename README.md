<!--
SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>

SPDX-License-Identifier: BSD-3-clause

This file is part of the bayesian-recurrence package
--->

# A Bayesian Interpretation of Recurrence in Neural Networks

This repository contains the different Bayesian recurrent units (BRUs)
implemented in PyTorch, that were defined in the following papers by A. Bittar
and P. Garner,
- [A Bayesian Interpretation of the Light Gated Recurrent Unit](https://rc.signalprocessingsociety.org/conferences/icassp-2021/SPSICASSP21VID0356.html?source=IBP), ICASSP 2021
- Bayesian Recurrent Units and the Forward-Backward Algorithm, INTERSPEECH 2022.


Contact: abittar@idiap.ch

## Installation via PyPi

Simply done with ``pip install bayesian-recurrence``

## Installation with GitHub

    git clone https://github.com/idiap/bayesian-recurrence.git
    cd bayesian-recurrence
    pip install -r requirements.txt

## Usage

After the installation, the defined recurrent units are available as python modules.
One can then create networks of the desired Bayesian units and use them inside PyTorch.

    
    import torch
    import torch.nn as nn
    
    from bayesian_recurrence import uBRU, liBRU, sBRU

    # Build input
    batch_size = 4
    nb_steps = 100
    nb_inputs = 20
    x = torch.Tensor(batch_size, nb_steps, nb_inputs)
    nn.init.uniform_(x)

    # Define network
    net = liBRU(
        nb_inputs,
        layer_sizes=[128, 128, 10],
        bidirectional=True,
        hidden_type='probs',
        normalization='batchnorm',
        use_bias=False,
        dropout=0.
        )

    # Pass input tensor through network
    y = net(x)
