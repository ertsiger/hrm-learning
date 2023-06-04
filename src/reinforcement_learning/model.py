import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from typing import Sequence


class MlpDQN(nn.Module):
    """
    Trivial MLP for testing the one-hot encoding.
    """
    def __init__(self, input_size, output_size):
        super(MlpDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class MinigridConv(nn.Module):
    """
    Convolutional neural network formed by 3 convolutional layers. This network is based on that of the following
    resources:
      - "Prioritized Level Replay" by Minqi Jiang, Edward Grefenstette, Tim Rocktäschel (2021).
        Code: https://github.com/facebookresearch/level-replay. The exact file is the following:
        https://github.com/facebookresearch/level-replay/blob/main/level_replay/model.py#L413
      - rl-starter-files GitHub repo (https://github.com/lcswillems/rl-starter-files). The exact file is the following:
        https://github.com/lcswillems/rl-starter-files/blob/master/model.py
    In the first source, they also used the 'full' observation provided by Minigrid although they don't use a DQN but
    PPO. The second source contains a very similar network to the one shown in the first, also used for PPO.
    """
    DEFAULT_IN_CHANNELS = 3
    KERNEL_SIZE = (2, 2)

    def __init__(self, obs_shape, num_out_channels=(16, 32, 32), use_max_pool=False):
        num_conv_layers = len(num_out_channels)

        assert (obs_shape[0] == MinigridConv.DEFAULT_IN_CHANNELS) or (obs_shape[0] == MinigridConv.DEFAULT_IN_CHANNELS - 1), \
            f"Error: Minigrid observations must consist of {MinigridConv.DEFAULT_IN_CHANNELS} matrices if colors are " \
            f"not removed, or {MinigridConv.DEFAULT_IN_CHANNELS - 1} if they are."
        assert num_conv_layers >= 1 and num_conv_layers <= 3, \
            "Error: The number of convolutional layers must be between 1 and 3."

        super(MinigridConv, self).__init__()

        self.conv1 = self._make_conv(obs_shape[0], num_out_channels[0])
        self.conv2 = self._make_conv(num_out_channels[0], num_out_channels[1]) if num_conv_layers >= 2 else None
        self.conv3 = self._make_conv(num_out_channels[1], num_out_channels[2]) if num_conv_layers == 3 else None
        self.use_max_pool = use_max_pool

        n, m = obs_shape[-2], obs_shape[-1]
        res_n, res_m, res_channels = n - 1, m - 1, num_out_channels[0]  # first convolution
        if self.use_max_pool:
            res_n, res_m = res_n // 2, res_m // 2
        if self.conv2 is not None:
            res_n, res_m, res_channels = res_n - 1, res_m - 1, num_out_channels[1]
        if self.conv3 is not None:
            res_n, res_m, res_channels = res_n - 1, res_m - 1, num_out_channels[2]
        self.embedding_size = res_n * res_m * res_channels

    def _make_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, MinigridConv.KERNEL_SIZE),
            nn.ReLU()
        )

    def get_embedding_size(self):
        return self.embedding_size

    def forward(self, obs):
        # Tip: follow advice in https://stackoverflow.com/questions/55667005/manage-memory-differently-on-train-and-test-time-pytorch
        out = self.conv1(obs)
        if self.use_max_pool:
            out = F.max_pool2d(out, MinigridConv.KERNEL_SIZE)
        if self.conv2 is not None:
            out = self.conv2(out)
        if self.conv3 is not None:
            out = self.conv3(out)
        out = out.flatten(1, -1)
        return out


def create_mlp(input_size, output_size, intermediate_outs):
    sequence = []
    curr_input = input_size
    for inter_out_size in intermediate_outs:
        sequence.append(nn.Linear(curr_input, inter_out_size))
        sequence.append(nn.ReLU())
        curr_input = inter_out_size
    sequence.append(nn.Linear(curr_input, output_size))
    return nn.Sequential(*sequence)


class MinigridFormulaDQN(nn.Module):
    """
    Network associated with the formulas on the edges of the automata in the hierarchies. Only used for the environments
    using Minigrid, i.e. CraftWorld.
    """
    def __init__(self, obs_shape, num_actions, conv_out_channels: Sequence[int], inter_lin_out: Sequence[int], use_max_pool):
        super(MinigridFormulaDQN, self).__init__()
        self.conv = MinigridConv(obs_shape, conv_out_channels, use_max_pool)
        self.mlp = create_mlp(self.conv.get_embedding_size(), num_actions, inter_lin_out)

    def forward(self, obs):
        out = self.conv(obs)
        out = self.mlp(out)
        return out


class MinigridMetaDQN(nn.Module):
    """
    Network associated with each of the automata in the hierarchy and used to determine which edge to take in a given
    automaton state. Only used for the environments using Minigrid, i.e. CraftWorld.
    """
    def __init__(self, obs_shape, num_automaton_states, num_observables, num_options, num_out_channels, use_max_pool):
        super(MinigridMetaDQN, self).__init__()
        self.conv = MinigridConv(obs_shape, num_out_channels, use_max_pool)
        self.mlp = create_mlp(self.conv.get_embedding_size() + num_automaton_states + num_observables, num_options, [64])

    def forward(self, obs, automaton_state_embedding, context_embedding):
        out = self.conv(obs)
        out = torch.cat((out, automaton_state_embedding, context_embedding), dim=1)
        out = self.mlp(out)
        return out


class MinigridDRQN(nn.Module):
    """
    DRQN network for Minigrid. The architecture is based on that used in our hierarchical approach: same convolutional
    network and LSTM layer with the same size as the linear layer used in our approach.
    """
    def __init__(self, obs_shape, num_observables, num_actions, lstm_method, hidden_size=256, use_max_pool=False):
        super(MinigridDRQN, self).__init__()

        self.obs_shape = obs_shape
        self.lstm_method = lstm_method
        self.hidden_size = hidden_size

        self.conv = MinigridConv(
            obs_shape,
            (16, 32, 32),
            use_max_pool
        )

        if lstm_method == "state":
            lstm_in_size = self.conv.get_embedding_size()
            linear_in_size = self.hidden_size
        elif lstm_method == "state+obs":
            lstm_in_size = self.conv.get_embedding_size() + num_observables
            linear_in_size = self.hidden_size
        elif lstm_method == "obs":
            lstm_in_size = num_observables
            linear_in_size = self.conv.get_embedding_size() + self.hidden_size
        else:
            raise RuntimeError(f"Error: Unknown method for using the LSTM memory '{lstm_method}'.")

        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_in_size,
            hidden_size=hidden_size
        )

        self.linear = nn.Linear(linear_in_size, num_actions)

    def get_zero_hidden_state(self, batch_size, device):
        return (
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device)
        )

    def forward(self, obs, observables, hidden_state):
        conv_out = self.conv(obs)

        if self.lstm_method == "state":
            lstm_hidden, lstm_cell = self._forward_lstm(conv_out, hidden_state)
            linear_in = lstm_hidden
        elif self.lstm_method == "state+obs":
            lstm_hidden, lstm_cell = self._forward_lstm(
                torch.concat((conv_out, observables), dim=1),
                hidden_state
            )
            linear_in = lstm_hidden
        elif self.lstm_method == "obs":
            lstm_hidden, lstm_cell = self._forward_lstm(
                observables, hidden_state
            )
            linear_in = torch.concat((conv_out, lstm_hidden), dim=1)
        else:
            raise RuntimeError(f"Error: Unknown method for using the LSTM memory '{self.lstm_method}'.")

        # Take the LSTM output of each step in the sequence and pass it to the linear layer to output the Q-values for
        # each step.
        q_vals = self.linear(linear_in)

        return q_vals, (lstm_hidden, lstm_cell)

    def _forward_lstm(self, in_seq, hidden_state):
        return self.lstm_cell(in_seq, hidden_state)


class MinigridCRMDQN(nn.Module):
    """
    CRM network for Minigrid. The architecture is based on that of our hierarchical approach for formulas: same
    convolutional network and a similar MLP (the only difference is that it takes a one-hot vector encoding the
    automaton state in which we want to estimate the values).
    """
    def __init__(self, obs_shape, num_automaton_states, num_actions, use_max_pool):
        super(MinigridCRMDQN, self).__init__()
        self.conv = MinigridConv(
            obs_shape,
            (16, 32, 32),
            use_max_pool
        )
        self.mlp = create_mlp(self.conv.get_embedding_size() + num_automaton_states, num_actions, (256,))

    def forward(self, obs, automaton_state_embdedding):
        out = self.conv(obs)
        out = torch.cat((out, automaton_state_embdedding), dim=1)
        out = self.mlp(out)
        return out


class MinigridDQN(nn.Module):
    """
    Regular DQN network for Minigrid. The architecture is based on that of our hierarchical approach for formulas: same
    convolutional network and a similar MLP (the difference is that the latter takes a vector encoding the observables
    seen at that step).
    """
    def __init__(self, obs_shape, num_observables, num_actions, use_max_pool=False):
        super(MinigridDQN, self).__init__()
        self.conv = MinigridConv(
            obs_shape,
            (16, 32, 32),
            use_max_pool
        )
        self.mlp = create_mlp(self.conv.get_embedding_size() + num_observables, num_actions, (256,))

    def forward(self, obs, observables):
        out = self.conv(obs)
        out = torch.cat((out, observables), dim=1)
        out = self.mlp(out)
        return out


class WaterWorldDQN(nn.Module):
    """
    Base DQN for WaterWorld used in several approaches, basically an MLP.
    """
    def __init__(self, input_size, output_size, inter_lin_out: Sequence[int]):
        super(WaterWorldDQN, self).__init__()
        self.mlp = create_mlp(input_size, output_size, inter_lin_out)

    def forward(self, input):
        return self.mlp(input)


class WaterWorldFormulaDQN(nn.Module):
    """
    Network associated with the formulas on the edges of the automata in the hierarchies. Used for WaterWorld.
    """
    def __init__(self, obs_shape, num_actions, inter_lin_out: Sequence[int]):
        super(WaterWorldFormulaDQN, self).__init__()
        self.dqn = WaterWorldDQN(obs_shape[0], num_actions, inter_lin_out)

    def forward(self, input):
        return self.dqn(input)


class WaterWorldMetaDQN(nn.Module):
    """
    Network associated with each of the automata in the hierarchy and used to determine which edge to take in a given
    automaton state. Used for WaterWorld.
    """
    def __init__(self, obs_shape, num_automaton_states, num_observables, num_options, inter_lin_out: Sequence[int]):
        super(WaterWorldMetaDQN, self).__init__()
        self.dqn = WaterWorldDQN(obs_shape[0] + num_automaton_states + num_observables, num_options, inter_lin_out)

    def forward(self, obs, automaton_state_embedding, context_embedding):
        obs_state_ctx = torch.cat((obs, automaton_state_embedding, context_embedding), dim=1)
        obs_mlp = self.dqn(obs_state_ctx)
        return obs_mlp


class WaterWorldDRQN(nn.Module):
    """
    DRQN network for WaterWorld. The architecture is based on that used in our hierarchical approach: same convolutional
    network and the last hidden layer is an LSTM layer with the same size as the linear layer used in our approach.
    """
    def __init__(self, obs_shape, output_size, inter_lin_out: Sequence[int] = (512, 512), hidden_size=512):
        super(WaterWorldDRQN, self).__init__()

        self.input_size = obs_shape[0]
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            *create_mlp(self.input_size, inter_lin_out[-1], inter_lin_out[:-1]),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=inter_lin_out[-1],
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def get_zero_hidden_state(self, batch_size, device):
        return (
            torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)
        )

    def forward(self, padded_obs_seq, hidden_state, unpadded_seq_lengths):
        batch_size, padded_length = padded_obs_seq.shape[0], padded_obs_seq.shape[1]

        # Reshape the (batch_size, seq_length, num_features) to (batch_size * seq_length, num_features) so that we can
        # easily apply the MLP.
        mlp_out = self.mlp(
            padded_obs_seq.reshape(batch_size * padded_length, self.input_size)
        )

        # Reshape the MLP output such that we have (batch_size, seq_length, num_features) and pack it using the original
        # sequence lengths (we assume the passed sequences were padded)
        lstm_in = pack_padded_sequence(
            mlp_out.reshape(batch_size, padded_length, -1),
            unpadded_seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequence to the LSTM.
        lstm_out, new_hidden_state = self.lstm(lstm_in, hidden_state)
        lstm_out_pad = pad_packed_sequence(lstm_out, batch_first=True, total_length=padded_length)[0]

        # Take the LSTM output of each step in the sequence and pass it to the linear layer to output the Q-values for
        # each step.
        q_vals = self.linear(lstm_out_pad)

        return q_vals, new_hidden_state


class WaterWorldCRMDQN(nn.Module):
    """
    CRM network for WaterWorld. The architecture is based on that of our hierarchical approach for formulas: same
    convolutional network and a similar MLP (the only difference is that it takes a one-hot vector encoding the
    automaton state in which we want to estimate the values).
    """
    def __init__(self, obs_shape, num_automaton_states, num_actions, layer_size=512):
        super(WaterWorldCRMDQN, self).__init__()
        self.dqn = WaterWorldDQN(obs_shape[0] + num_automaton_states, num_actions, (layer_size, layer_size, layer_size))

    def forward(self, obs, automaton_state_embdedding):
        return self.dqn(torch.cat((obs, automaton_state_embdedding), dim=1))


class WaterWorldRegularDQN(nn.Module):
    """
    Regular DQN network for WaterWorld. The architecture is based on that of our hierarchical approach for formulas:
    same convolutional network and a similar MLP (the difference is that the latter takes a vector encoding the
    observables seen at that step).
    """
    def __init__(self, obs_shape, num_observables, num_actions):
        super(WaterWorldRegularDQN, self).__init__()
        self.dqn = WaterWorldDQN(obs_shape[0] + num_observables, num_actions, (512, 512, 512))

    def forward(self, obs, observables):
        out = torch.cat((obs, observables), dim=1)
        out = self.mlp(out)
        return out
