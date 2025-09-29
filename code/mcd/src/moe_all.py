import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import h5py
import os
import torch.nn.functional as F

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.soft(out)
        return out


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        model=MLP,
        noisy_gating=True,
        k=4,
        all_mode=False,
    ):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        h5_path = "/data/zhangruwen/MM_Controversy_Detection_Released-main/" \
                  "predataset/text_result/ours_task_feature_clip.h5"
        with h5py.File(h5_path, "r") as f:
            self.task_matrix = f["task_embedding"][:]  # 存的名字是 "task_embedding"
        self.num_tasks, self.task_emb_dim = self.task_matrix.shape
        self.task_emb_dim2 = 64

        # === MOE初始化 ===
        # self.num_tasks = 2
        # self.task_emb_dim = 8
        self.task_embedding = nn.Linear(self.task_emb_dim, self.task_emb_dim2)
        self.gate_input_dim = input_size + self.task_emb_dim2
        # instantiate experts
        self.experts = nn.ModuleList(
            [
                model(self.input_size, self.output_size, self.hidden_size)
                for i in range(self.num_experts)
            ]
        )
        self.w_gate = nn.Parameter(
            torch.zeros(self.gate_input_dim, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(self.gate_input_dim, num_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.1]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

        # # === 1. PLE专家网络定义（共享 + 每任务）===
        # self.num_shared_experts = num_experts // 2
        # self.num_task_experts = num_experts // 2
        #
        # self.shared_experts = nn.ModuleList([
        #     model(self.input_size, self.hidden_size, self.hidden_size)
        #     for _ in range(self.num_shared_experts)
        # ])
        #
        # self.task_experts = nn.ModuleDict({
        #     'task_0': nn.ModuleList([
        #         model(self.input_size, self.hidden_size, self.hidden_size)
        #         for _ in range(self.num_task_experts)
        #     ]),
        #     'task_1': nn.ModuleList([
        #         model(self.input_size, self.hidden_size, self.hidden_size)
        #         for _ in range(self.num_task_experts)
        #     ])
        # })
        #
        # # === 2. 门控网络（每个任务一个门控）===
        # self.gates = nn.ModuleDict({
        #     'task_0': nn.Linear(self.input_size, self.num_shared_experts + self.num_task_experts),
        #     'task_1': nn.Linear(self.input_size, self.num_shared_experts + self.num_task_experts),
        # })
        #
        # # === 3. 每个任务的预测头 ===
        # self.towers = nn.ModuleDict({
        #     'task_0': nn.Sequential(
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(self.hidden_size, self.output_size)  # 分类：output_size=类别数
        #     ),
        #     'task_1': nn.Sequential(
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(self.hidden_size, self.output_size)  # 回归任务
        #     ),
        # })

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward_PLE(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, input_size]
        Returns:
            out_task_0: classification output
            out_task_1: regression output
        """

        # === 1. 获取共享专家输出 ===
        shared_outputs = [expert(x) for expert in self.shared_experts]  # list of [batch, hidden]
        shared_outputs = torch.stack(shared_outputs, dim=1)  # shape: [batch, num_shared, hidden]

        # === 2. 每任务的专家输出 + 拼接共享专家 ===
        task_outputs = {}
        for task_name in ['task_0', 'task_1']:
            # 每任务专家输出
            specific_experts = [expert(x) for expert in self.task_experts[task_name]]
            specific_experts = torch.stack(specific_experts, dim=1)  # [batch, num_task, hidden]

            # 拼接共享 + 专属专家输出
            all_expert_outputs = torch.cat([specific_experts, shared_outputs], dim=1)  # [batch, total_expert, hidden]

            # 门控权重
            gate_logits = self.gates[task_name](x)  # [batch, total_expert]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, total_expert]
            gate_weights = gate_weights.unsqueeze(-1)  # [batch, total_expert, 1]

            # 门控加权融合专家输出
            task_input = torch.sum(all_expert_outputs * gate_weights, dim=1)  # [batch, hidden]

            task_outputs[task_name] = task_input

        # === 3. 每任务预测输出 ===
        out_task_0 = self.towers['task_0'](task_outputs['task_0'])  # 分类
        out_task_1 = self.towers['task_1'](task_outputs['task_1'])  # 回归

        return out_task_0, out_task_1

    def forward(self, x, loss_coef=1e-2):  # 任务感知的
        """
        Args:
            x: tensor shape [batch_size, input_size]
            loss_coef: scalar - load-balancing loss multiplier
        Returns:
            (y_a, y_b): outputs for task A and task B
            loss: auxiliary load-balancing loss (sum of both tasks)
        """

        # Task embeddings
        # task_emb_A = self.task_embeddings[0].unsqueeze(0).expand(x.size(0), -1)  # [batch_size, task_dim]
        # task_emb_B = self.task_embeddings[1].unsqueeze(0).expand(x.size(0), -1)

        task_emb_A = torch.FloatTensor(self.task_matrix[0, :]).unsqueeze(0).to(x.device)
        task_emb_B = torch.FloatTensor(self.task_matrix[1, :]).unsqueeze(0).to(x.device)
        task_emb_A = self.task_embedding(task_emb_A)  # if task_embedding2 is a linear layer
        task_emb_B = self.task_embedding(task_emb_B)
        task_emb_A = task_emb_A.expand(x.size(0), -1)  # shape: [batch_size, task_dim]
        task_emb_B = task_emb_B.expand(x.size(0), -1)


        # Augment input with task embedding
        x_a = torch.cat([x, task_emb_A], dim=-1)  # for task A
        x_b = torch.cat([x, task_emb_B], dim=-1)  # for task B

        # === Task A classify===
        gates_a, load_a = self.noisy_top_k_gating(x_a, self.training)
        importance_a = gates_a.sum(0)
        loss_a = self.cv_squared(importance_a) + self.cv_squared(load_a)
        loss_a *= loss_coef

        dispatcher_a = SparseDispatcher(self.num_experts, gates_a)
        expert_inputs_a = dispatcher_a.dispatch(x)
        expert_outputs_a = [self.experts[i](expert_inputs_a[i]) for i in range(self.num_experts)]
        y_a = dispatcher_a.combine(expert_outputs_a)

        # === Task B regression===
        gates_b, load_b = self.noisy_top_k_gating(x_b, self.training)
        importance_b = gates_b.sum(0)
        loss_b = self.cv_squared(importance_b) + self.cv_squared(load_b)
        loss_b *= loss_coef

        dispatcher_b = SparseDispatcher(self.num_experts, gates_b)
        expert_inputs_b = dispatcher_b.dispatch(x)
        expert_outputs_b = [self.experts[i](expert_inputs_b[i]) for i in range(self.num_experts)]
        y_b = dispatcher_b.combine(expert_outputs_b)

        # Return both task outputs and combined loss
        return y_a, y_b, loss_a, loss_b

    def forward_raw(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss

    def forward_roberta(self, tweets_tensor, des_tensor, loss_coef=1e-2):
        # not sure
        x = des_tensor
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        x = torch.cat([tweets_tensor, des_tensor], dim=-1)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](
                expert_inputs[i][:, : tweets_tensor.shape[1]],
                expert_inputs[i][:, tweets_tensor.shape[1] :],
            )
            for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class MoE_share_gate(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        model=MLP,
        noisy_gating=True,
        k=4,
        all_mode=False,
    ):
        super(MoE_share_gate, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList(
            [
                model(self.input_size, self.output_size, self.hidden_size)
                for i in range(self.num_experts)
            ]
        )
        self.w_gate = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        gates_out = gates
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss, gates_out


class MoE_receive_gate(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        model=MLP,
        noisy_gating=True,
        k=4,
        all_mode=False,
        special_for_gate=256,
    ):
        super(MoE_receive_gate, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.w_gate = nn.Parameter(
            torch.zeros(special_for_gate, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(special_for_gate, num_experts), requires_grad=True
        )
        self.experts = nn.ModuleList(
            [
                model(self.input_size, self.output_size, self.hidden_size)
                for i in range(self.num_experts)
            ]
        )
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, gate, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        gates, load = self.noisy_top_k_gating(gate, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss

    def forward_roberta(self, tweets_tensor, des_tensor, gate, loss_coef=1e-2):
        # not sure
        gates, load = self.noisy_top_k_gating(gate, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        x = torch.cat([tweets_tensor, des_tensor], dim=-1)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](
                expert_inputs[i][:, : tweets_tensor.shape[1]],
                expert_inputs[i][:, tweets_tensor.shape[1] :],
            )
            for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss
