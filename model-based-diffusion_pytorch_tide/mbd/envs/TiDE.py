import numpy as np
import pandas as pd
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pickle import dump
from sklearn.preprocessing import MinMaxScaler
import time

# from pytorch_forecasting.metrics.quantile import QuantileLoss

# For LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

# For TFT
from tqdm import tqdm_notebook as tqdm

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

# ==================================
# Forward Propagation
# ==================================
def TiDE_forward(u_hat:np.array, # future u values within the control horizon, length = M; this should be the warm start versionde
        u_past:np.array, # past u
        x_past: np.array, # past state of x, size (N,window)
        SP_hat:np.array, # Reference trajectory, length = P
        P, # Predictive Horizon
        TiDE # TiDE model that we're using
):

    # convert u_hat into tensor and set u_hat as variable
    u_hat = torch.tensor(u_hat.reshape(-1,1), requires_grad=False, dtype=torch.float32)
    u_hat_in = u_hat.unsqueeze(0)

    # knit past and future covariate into the input format for TiDE
    past_cov = torch.tensor(np.concatenate((x_past,u_past), axis = 0),dtype=torch.float32).transpose(1,0).unsqueeze(0)
    # TiDE prediction
    x_hat = TiDE([past_cov,u_hat_in,None])


    # compute objective value
    return  x_hat[0,:,:,1], x_hat
    #return  x_hat[0,:,:,0]

# ======================================================================================
# Quantile loss for time series from Darts
# ======================================================================================

def quantile_loss(model_output: torch.Tensor, target: torch.Tensor, quantiles):
        """
        We are re-defining a custom loss (which is not a likelihood loss) compared to Likelihood

        Parameters
        ----------
        model_output
            must be of shape (batch_size, n_timesteps, n_target_variables, n_quantiles)
        target
            must be of shape (n_samples, n_timesteps, n_target_variables)
        quantiles
            a tensor of quantiles
        """

        dim_q = 3

        batch_size, length = model_output.shape[:2]
        device = model_output.device


        quantiles_tensor = torch.tensor(quantiles).to(device)

        errors = target.unsqueeze(-1) - model_output
        losses = torch.max(
            (quantiles_tensor - 1) * errors, quantiles_tensor * errors
        )

        return losses.sum(dim=dim_q).mean()



# ================================================================
# TiDE
# ================================================================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Pytorch module implementing the Residual Block from the TiDE paper."""
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # residual connection
        x = self.dense(x) + self.skip(x)

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x
    


class TideModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        input_chunck_length: int,
        output_chunk_length: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_past: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        **kwargs,
    ):
        """Pytorch module implementing the TiDE architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional past covariates + optional future covariates).
        output_dim
            Number of output components in the target.
        future_cov_dim
            Number of future covariates.
        static_cov_dim
            Number of static covariates.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        num_encoder_layers
            Number of stacked Residual Blocks in the encoder.
        num_decoder_layers
            Number of stacked Residual Blocks in the decoder.
        decoder_output_dim
            The number of output components of the decoder.
        hidden_size
            The width of the hidden layers in the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The width of the hidden layers in the temporal decoder.
        temporal_width_past (L)
            The width of the past covariate embedding space.
        temporal_width_future (H)
            The width of the future covariate embedding space.
        use_layer_norm
            Whether to use layer normalization in the Residual Blocks.
        dropout
            Dropout probability
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x
            Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Outputs
        -------
        y
            Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`

        """

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.input_chunk_length = input_chunck_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future

        # past covariates handling: either feature projection, raw features, or no features
        self.past_cov_projection = None
        if self.past_cov_dim and temporal_width_past:
            # residual block for past covariates feature projection
            self.past_cov_projection = ResidualBlock(
                input_dim=self.past_cov_dim,
                output_dim=temporal_width_past,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            past_covariates_flat_dim = self.input_chunk_length * temporal_width_past
        elif self.past_cov_dim:
            # skip projection and use raw features
            past_covariates_flat_dim = self.input_chunk_length * self.past_cov_dim
        else:
            past_covariates_flat_dim = 0

        # future covariates handling: either feature projection, raw features, or no features
        self.future_cov_projection = None
        if future_cov_dim and self.temporal_width_future:
            # residual block for future covariates feature projection
            self.future_cov_projection = ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * temporal_width_future
        elif future_cov_dim:
            # skip projection and use raw features
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0

        encoder_dim = (
            self.input_chunk_length * output_dim
            + past_covariates_flat_dim
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim
                * self.output_chunk_length
                * self.nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        decoder_input_dim = decoder_output_dim * self.nr_params
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length * self.nr_params
        )


    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TiDE model forward pass.
        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Returns
        -------
        torch.Tensor
            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
        """

        # x has shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, input_chunk_length, future_cov_dim) # or output_chunk_length?
        # x_static_covariates has shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        # x_lookback is the y_lookback in the paper. shape (batch_size,input_chunk_length,output_dim). output_dim here is the dimension of y
        x_lookback = x[:, :, : self.output_dim] 

        # future covariates: feature projection or raw features
        # historical future covariates need to be extracted from x and stacked with part of future covariates
        if self.future_cov_dim: # if given future covariate
            x_dynamic_future_covariates = torch.cat(
                [x[:,:,None if self.future_cov_dim == 0 else -self.future_cov_dim :,], x_future_covariates,],
                dim=1,
            ) # shape (batch, input_chunck_length + output_chunck_length, future_cov_dim)
            if self.temporal_width_future:
                # project input features across all input and output time steps; feed them into the feature projection residual block. 
                # the output shape should be (batchj, L+H, temporal_width_future)
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                ) 
        else:
            x_dynamic_future_covariates = None

        # past covariates: feature projection or raw features
        # the past covariates are embedded in `x`
        # The reason why we separate past covariate and future covariate is because some of the past covariates may not be provided as a future covariates    
        if self.past_cov_dim:
            x_dynamic_past_covariates = x[:,:,self.output_dim : self.output_dim + self.past_cov_dim,]
            if self.temporal_width_past:
                # project input features across all input time steps
                x_dynamic_past_covariates = self.past_cov_projection(
                    x_dynamic_past_covariates
                )
        else:
            x_dynamic_past_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,                       # (batch_size,input_chunk_length,output_dim)
            x_dynamic_past_covariates,        # None
            x_dynamic_future_covariates,      # (batch_size_, L+H, future_cov_dim)
            x_static_covariates,              # None
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, decode, reshape
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        # get view that is batch size x output chunk length x self.decoder_output_dim x nr params
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2) # shape = (batch_size, H, decoder_out_dim*nr_param + future_cov_dim/temporal_width_dim)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input) # shape (batch_size, H, output_dim*n_param)


        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return y
    


        