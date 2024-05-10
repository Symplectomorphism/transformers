"""
Sparsely-Gated Mixture-of-Experts from `Outrageously Large Neural Networks`
https://arxiv.org/abs/1701.06538
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mlp import MLP
import numpy as np


class SparseDispatcher(object):
    """
    Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the experts and
    to combine the results of the experts to form a unified output tensor.

    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a unified output
    Tensor. Outputs from different experts for the same batch element are summed
    together, weighted by the provided `gates`.

    The class is initialized with a `gates` Tensor, which specifies which batch 
    elements go to which experts, and the weights to use when combining the 
    outputs. Batch element b is sent to expert e iff gates[b, e] != 0. 

    The inputs and outputs are all two-dimensional [batch, depth]. Caller is 
    responsible for collapsing additional dimensions prior to calling this 
    class and reshaping the output to the original shape.
    See common_layers.reshape_like().

    Example use:
    ------------
    gates: a float32 tensor with shape [batch_size, num_experts],
    inputs: a float32 tensor with shape [batch_size, input_size],
    experts: a list of length num_experts containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)

    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))

    This class takes advantage of the sparsity in the gate matrix by including
    in the tensors for expert i only the batch elements for which gates[b, i] 
    > 0.
    """

    def __init__(self, num_experts, gates):
        """ Create a SparseDispatcher. """

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according to batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, input):
        """ Create one input tensor for each expert.
        The tensor for expert i contains slices of the input corresponding to 
        the batch elements b where gates[b, i] > 0.
        Args:
          input: a tensor of shape [batch_size, <extra_input_dims>]
        Returns:
          a list of num_experts tensors with shapes
          [expert_batch_size_i, <extra_input_dims>]
        """
        # expand according to batch index so we can split by _part_sizes
        input_exp = input[self._batch_index].squeeze(1)
        return torch.split(input_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """ Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element b is computed as 
        the sum over all experts i of the expert output, weighted by the 
        corresponding gate values. If `multiply_by_gates` is set to False, the 
        gate values are ignored.
        Args:
          expert_out: a list of `num_perts` tensors each with shape
            [expert_batch_size_i, <extra_output_dims>]
          multiply_by_gates: a boolean
        Returns: 
          a tensor with shape [batch_size, <extra_output_dims>]
        """
        # apply exp to expert outputs so we are no longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined==0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert tensors.
        Returns:
          a list of num_experts one-dimensional tensors with type tf.float32 and
          shapes [expert_batch-size_i]
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    """ Call a sparsely gated mixture-of-experts layer with 1-layer feedforward 
    networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the output
    num_experts: integer - number of experts
    hidden_size: integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - number of experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons are added for numerical stability.
        Returns 0 for an empty tensor.
        Args:
          x: a tensor
        Returns:
          a scalar tensor
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
          gates: a tensor of shape [batch_size, n]
        Returns:
          a tensor of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """ Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number 
        of times each expert is in the top k experts per sample.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable!
        Args:
          clean_values: a tensor of shape [batch, n]
          noisy_values: a tensor of shape [batch, n]. Equal to clean values plus 
            normally distributed noise with standard deviation noise_stddev
          noise_stddev: a tensor of shape [batch, n] or None
          noisy_top_values: a tensor of shape [batch, m]
            "values" output of torch.topk(noisy_top_values, m); m >= k+1
        Returns:
          a tensor of shape [batch, n] of probabilities
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """ Noisy top-k gating
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input of shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a tensor of shape [batch_size, num_experts]
          load: a tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(k=min(self.k+1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, train=True, loss_coeff=1e-2):
        """
        Args:
          x: input of shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          loss_coeff: a float - multiplier on load-balancing losses
        Returns:
          y: a tensor of shape [batch_size, output_size]
          extra_training_loss: a float - this should be added into the overall 
          training loss of the model. The backprop of this loss encourages all 
          experts to be approximatly equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, train)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coeff

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss