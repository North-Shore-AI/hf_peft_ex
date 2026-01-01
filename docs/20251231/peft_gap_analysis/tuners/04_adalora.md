# AdaLoRA (Adaptive LoRA)

## Overview

AdaLoRA dynamically allocates rank budget across layers during training based on importance scores. Layers with higher importance get more rank, while less important layers get pruned.

## Python Reference

**Files:**
- `peft/src/peft/tuners/adalora/config.py` (~100 lines)
- `peft/src/peft/tuners/adalora/layer.py` (~400 lines)
- `peft/src/peft/tuners/adalora/model.py` (~350 lines)
- `peft/src/peft/tuners/adalora/bnb.py` (~150 lines)
- `peft/src/peft/tuners/adalora/gptq.py` (~80 lines)

### AdaLoraConfig

```python
@dataclass
class AdaLoraConfig(LoraConfig):
    target_r: int = 8          # Target average rank
    init_r: int = 12           # Initial rank (higher, then pruned)
    tinit: int = 0             # Warmup steps before pruning
    tfinal: int = 0            # Steps to freeze ranks
    deltaT: int = 1            # Pruning interval
    beta1: float = 0.85        # EMA for sensitivity
    beta2: float = 0.85        # EMA for uncertainty
    orth_reg_weight: float = 0.5  # Orthogonal regularization
    total_step: int = None     # Total training steps (required)
    rank_pattern: dict = None  # Final rank per layer (set during training)
```

### AdaLoraLayer

```python
class AdaLoraLayer(LoraLayer):
    # SVD parameterization: W = W0 + P @ diag(Lambda) @ Q^T
    lora_A: dict  # Q matrices (same as LoRA A)
    lora_B: dict  # P matrices (same as LoRA B)
    lora_E: dict  # Lambda (singular values) - PRUNABLE
    ranknum: dict # Current rank per adapter

    def __init__(self, ...):
        # Initialize with init_r rank
        self.r[adapter_name] = init_r
        self.ranknum[adapter_name] = init_r

    def update_layer(self, adapter_name, r, lora_alpha, ...):
        # P matrix (like lora_B but transposed role)
        self.lora_A[adapter_name] = nn.Parameter(torch.zeros(r, in_features))
        # Q matrix (like lora_A but transposed role)
        self.lora_B[adapter_name] = nn.Parameter(torch.zeros(out_features, r))
        # Singular values (what gets pruned)
        self.lora_E[adapter_name] = nn.Parameter(torch.ones(r, 1))
        # Rank tracking
        self.ranknum[adapter_name] = r

    def forward(self, x):
        # Base output
        result = self.base_layer(x)

        if self.disable_adapters:
            return result

        for adapter in self.active_adapters:
            A = self.lora_A[adapter]  # Q
            B = self.lora_B[adapter]  # P
            E = self.lora_E[adapter]  # Lambda

            # Delta = P @ diag(Lambda) @ Q^T
            # Computed as: B @ (E * A)
            delta = x @ (E * A).T @ B.T * self.scaling[adapter]
            result = result + delta

        return result
```

### RankAllocator

```python
class RankAllocator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.ipt = {}     # Importance scores
        self.exp_avg_ipt = {}  # EMA of importance
        self.exp_avg_unc = {}  # EMA of uncertainty

    def compute_ipt(self, model):
        """Compute importance = |gradient * parameter|"""
        for name, param in model.named_parameters():
            if "lora_E" in name:
                ipt = (param * param.grad).abs().detach()
                self.ipt[name] = ipt

    def update_ema(self, name, ipt):
        """Update exponential moving average"""
        self.exp_avg_ipt[name] = (
            self.config.beta1 * self.exp_avg_ipt.get(name, 0) +
            (1 - self.config.beta1) * ipt
        )

    def mask_to_budget(self, model, budget):
        """Prune singular values to meet budget"""
        # Collect all importance scores
        all_ipt = []
        for name, layer in get_adalora_layers(model):
            all_ipt.append((layer.lora_E[adapter], self.exp_avg_ipt[name]))

        # Sort and find threshold
        threshold = compute_threshold(all_ipt, budget)

        # Mask values below threshold
        for name, layer in get_adalora_layers(model):
            mask = self.exp_avg_ipt[name] >= threshold
            layer.ranknum[adapter] = mask.sum().item()
            # Zero out pruned singular values
            layer.lora_E[adapter].data *= mask

    def update_and_allocate(self, model, global_step):
        """Called each training step"""
        if global_step < self.config.tinit:
            # Warmup: no pruning
            return

        if global_step > self.config.total_step - self.config.tfinal:
            # Final: freeze ranks
            return

        if global_step % self.config.deltaT == 0:
            # Compute importance
            self.compute_ipt(model)
            self.update_ema(...)

            # Calculate current budget
            budget = self.get_budget(global_step)

            # Prune to budget
            self.mask_to_budget(model, budget)

    def get_budget(self, step):
        """Linear schedule from init_r to target_r"""
        progress = (step - self.config.tinit) / (
            self.config.total_step - self.config.tfinal - self.config.tinit
        )
        return self.config.init_r - progress * (self.config.init_r - self.config.target_r)
```

## Elixir Implementation Design

### Module: `HfPeftEx.Tuners.Adalora.Config`

```elixir
defmodule HfPeftEx.Tuners.Adalora.Config do
  @moduledoc """
  Configuration for AdaLoRA with adaptive rank allocation.
  """

  @derive Jason.Encoder
  defstruct [
    # Base LoRA config fields
    peft_type: :adalora,
    task_type: nil,
    base_model_name_or_path: nil,
    inference_mode: false,
    r: 8,
    lora_alpha: 8,
    lora_dropout: 0.0,
    target_modules: nil,
    exclude_modules: nil,
    fan_in_fan_out: false,
    bias: :none,
    modules_to_save: nil,
    # AdaLoRA specific
    target_r: 8,           # Target average rank
    init_r: 12,            # Initial rank
    tinit: 0,              # Warmup steps
    tfinal: 0,             # Final freeze steps
    delta_t: 1,            # Pruning interval
    beta1: 0.85,           # EMA coefficient for importance
    beta2: 0.85,           # EMA coefficient for uncertainty
    orth_reg_weight: 0.5,  # Orthogonal regularization weight
    total_step: nil,       # Required: total training steps
    rank_pattern: nil      # Saved rank pattern after training
  ]

  @type t :: %__MODULE__{}

  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct(__MODULE__, opts)
    |> validate!()
  end

  defp validate!(config) do
    if is_nil(config.total_step) or config.total_step <= 0 do
      raise ArgumentError, "total_step is required and must be positive"
    end

    if config.tinit >= config.total_step - config.tfinal do
      raise ArgumentError, "tinit must be less than total_step - tfinal"
    end

    if config.init_r < config.target_r do
      raise ArgumentError, "init_r must be >= target_r"
    end

    config
  end
end
```

### Module: `HfPeftEx.Tuners.Adalora.Layer`

```elixir
defmodule HfPeftEx.Tuners.Adalora.Layer do
  @moduledoc """
  AdaLoRA layer with SVD parameterization and prunable singular values.
  """

  import Nx.Defn

  defstruct [
    :base_layer,
    :in_features,
    :out_features,
    lora_a: %{},      # Q matrices (r x in_features)
    lora_b: %{},      # P matrices (out_features x r)
    lora_e: %{},      # Lambda singular values (r x 1)
    ranknum: %{},     # Current effective rank
    scaling: %{},
    active_adapter: "default",
    merged_adapters: [],
    disable_adapters: false,
    merged: false
  ]

  @type t :: %__MODULE__{}

  @doc """
  Create a new AdaLoRA layer.
  """
  @spec new(map(), keyword()) :: t()
  def new(base_layer, opts \\ []) do
    {out_features, in_features} = Nx.shape(base_layer.weight)
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)

    {in_features, out_features} = if fan_in_fan_out do
      {out_features, in_features}
    else
      {in_features, out_features}
    end

    %__MODULE__{
      base_layer: base_layer,
      in_features: in_features,
      out_features: out_features
    }
  end

  @doc """
  Add an adapter with SVD parameterization.
  """
  @spec update_layer(t(), String.t(), non_neg_integer(), number()) :: t()
  def update_layer(layer, adapter_name, r, lora_alpha) do
    # Q matrix (like LoRA A)
    lora_a = Nx.random_normal({r, layer.in_features}, 0.0, 0.02)

    # P matrix (like LoRA B) - initialized to small values
    lora_b = Nx.broadcast(0.0, {layer.out_features, r})

    # Singular values - initialized to ones
    lora_e = Nx.broadcast(1.0, {r, 1})

    scaling = lora_alpha / r

    layer
    |> put_in([:lora_a, adapter_name], lora_a)
    |> put_in([:lora_b, adapter_name], lora_b)
    |> put_in([:lora_e, adapter_name], lora_e)
    |> put_in([:ranknum, adapter_name], r)
    |> put_in([:scaling, adapter_name], scaling)
  end

  @doc """
  Forward pass with SVD decomposition.
  """
  defn forward(layer, x) do
    base_output = apply_base_layer(layer.base_layer, x)

    if layer.disable_adapters or layer.merged do
      base_output
    else
      adapter = layer.active_adapter
      a = layer.lora_a[adapter]  # Q: (r, in)
      b = layer.lora_b[adapter]  # P: (out, r)
      e = layer.lora_e[adapter]  # Lambda: (r, 1)
      scaling = layer.scaling[adapter]

      # Delta = x @ (E * A)^T @ B^T = x @ A^T @ diag(E) @ B^T
      # Reshape E for broadcasting
      e_diag = Nx.squeeze(e)  # (r,)

      # Compute: x @ A^T gives (batch, r)
      # Then multiply by E element-wise
      # Then multiply by B^T
      xa = Nx.dot(x, Nx.transpose(a))  # (batch, r)
      xa_e = xa * e_diag  # (batch, r)
      delta = Nx.dot(xa_e, Nx.transpose(b)) * scaling  # (batch, out)

      base_output + delta
    end
  end

  @doc """
  Get delta weight for merging: B @ diag(E) @ A
  """
  @spec get_delta_weight(t(), String.t()) :: Nx.Tensor.t()
  def get_delta_weight(layer, adapter) do
    a = layer.lora_a[adapter]
    b = layer.lora_b[adapter]
    e = Nx.squeeze(layer.lora_e[adapter])
    scaling = layer.scaling[adapter]

    # B @ diag(E) @ A = B @ (E .* A)
    ea = e * a
    Nx.dot(b, ea) |> Nx.multiply(scaling)
  end

  @doc """
  Update singular values with mask (for pruning).
  """
  @spec apply_mask(t(), String.t(), Nx.Tensor.t()) :: t()
  def apply_mask(layer, adapter, mask) do
    new_e = Nx.multiply(layer.lora_e[adapter], mask)
    new_rank = Nx.sum(mask) |> Nx.to_number() |> trunc()

    layer
    |> put_in([:lora_e, adapter], new_e)
    |> put_in([:ranknum, adapter], new_rank)
  end

  defp apply_base_layer(base_layer, x) do
    Nx.dot(x, Nx.transpose(base_layer.weight))
    |> maybe_add_bias(base_layer)
  end

  defp maybe_add_bias(output, %{bias: nil}), do: output
  defp maybe_add_bias(output, %{bias: bias}), do: Nx.add(output, bias)
end
```

### Module: `HfPeftEx.Tuners.Adalora.RankAllocator`

```elixir
defmodule HfPeftEx.Tuners.Adalora.RankAllocator do
  @moduledoc """
  Manages adaptive rank allocation during AdaLoRA training.
  """

  defstruct [
    :config,
    :adapter_name,
    ipt: %{},          # Current importance scores
    exp_avg_ipt: %{},  # EMA of importance
    exp_avg_unc: %{}   # EMA of uncertainty
  ]

  @type t :: %__MODULE__{}

  @spec new(struct(), String.t()) :: t()
  def new(config, adapter_name \\ "default") do
    %__MODULE__{
      config: config,
      adapter_name: adapter_name
    }
  end

  @doc """
  Compute importance scores from gradients.
  """
  @spec compute_importance(t(), map()) :: t()
  def compute_importance(allocator, gradients) do
    new_ipt = for {name, %{lora_e: e, lora_e_grad: grad}} <- gradients, into: %{} do
      # Importance = |param * gradient|
      ipt = Nx.abs(Nx.multiply(e, grad))
      {name, ipt}
    end

    %{allocator | ipt: new_ipt}
  end

  @doc """
  Update exponential moving averages.
  """
  @spec update_ema(t()) :: t()
  def update_ema(allocator) do
    beta1 = allocator.config.beta1
    beta2 = allocator.config.beta2

    new_exp_avg_ipt = for {name, ipt} <- allocator.ipt, into: %{} do
      prev = Map.get(allocator.exp_avg_ipt, name, Nx.broadcast(0.0, Nx.shape(ipt)))
      new_avg = Nx.add(Nx.multiply(prev, beta1), Nx.multiply(ipt, 1 - beta1))
      {name, new_avg}
    end

    new_exp_avg_unc = for {name, ipt} <- allocator.ipt, into: %{} do
      prev_avg = Map.get(allocator.exp_avg_ipt, name, ipt)
      prev_unc = Map.get(allocator.exp_avg_unc, name, Nx.broadcast(0.0, Nx.shape(ipt)))
      # Uncertainty as |current - average|
      unc = Nx.abs(Nx.subtract(ipt, prev_avg))
      new_unc = Nx.add(Nx.multiply(prev_unc, beta2), Nx.multiply(unc, 1 - beta2))
      {name, new_unc}
    end

    %{allocator | exp_avg_ipt: new_exp_avg_ipt, exp_avg_unc: new_exp_avg_unc}
  end

  @doc """
  Get current budget based on training step.
  """
  @spec get_budget(t(), non_neg_integer()) :: float()
  def get_budget(allocator, global_step) do
    config = allocator.config
    tinit = config.tinit
    tfinal = config.tfinal
    total = config.total_step

    if global_step < tinit do
      config.init_r * 1.0
    else
      progress = (global_step - tinit) / (total - tfinal - tinit)
      config.init_r - progress * (config.init_r - config.target_r)
    end
  end

  @doc """
  Compute masks for each layer to meet budget.
  """
  @spec compute_masks(t(), float()) :: %{String.t() => Nx.Tensor.t()}
  def compute_masks(allocator, budget) do
    # Collect all importance scores with layer names
    all_scores = for {name, scores} <- allocator.exp_avg_ipt do
      unc = Map.get(allocator.exp_avg_unc, name, scores)
      # Combined score (importance + uncertainty for exploration)
      combined = Nx.add(scores, unc)
      {name, combined}
    end

    # Flatten and find threshold
    all_values = Enum.flat_map(all_scores, fn {_, t} ->
      Nx.to_flat_list(t)
    end)

    total_params = length(all_values)
    keep_count = trunc(budget * total_params / allocator.config.init_r)

    sorted = Enum.sort(all_values, :desc)
    threshold = Enum.at(sorted, min(keep_count - 1, length(sorted) - 1), 0.0)

    # Generate masks
    for {name, scores} <- all_scores, into: %{} do
      mask = Nx.greater_equal(scores, threshold)
      {name, mask}
    end
  end

  @doc """
  Main update function called each training step.
  """
  @spec update_and_allocate(t(), map(), non_neg_integer()) :: {t(), %{String.t() => Nx.Tensor.t()} | nil}
  def update_and_allocate(allocator, gradients, global_step) do
    config = allocator.config

    cond do
      # Warmup phase: no pruning
      global_step < config.tinit ->
        {allocator, nil}

      # Final phase: freeze ranks
      global_step > config.total_step - config.tfinal ->
        {allocator, nil}

      # Pruning interval
      rem(global_step, config.delta_t) == 0 ->
        allocator = allocator
        |> compute_importance(gradients)
        |> update_ema()

        budget = get_budget(allocator, global_step)
        masks = compute_masks(allocator, budget)

        {allocator, masks}

      true ->
        {allocator, nil}
    end
  end
end
```

## Files to Read

**Python (required reading):**
- `peft/src/peft/tuners/adalora/config.py`
- `peft/src/peft/tuners/adalora/layer.py`
- `peft/src/peft/tuners/adalora/model.py`

**Elixir (context):**
- `lib/hf_peft_ex/tuners/lora/layer.ex`
- `lib/hf_peft_ex/tuners/lora/config.ex`

## Tests Required

1. **Config Tests:**
   - Validate total_step required
   - Validate tinit < total_step - tfinal
   - Validate init_r >= target_r

2. **Layer Tests:**
   - SVD parameterization initialization
   - Forward with E scaling
   - Delta weight computation
   - Mask application (pruning)

3. **RankAllocator Tests:**
   - Importance computation
   - EMA updates
   - Budget scheduling
   - Mask computation
   - Warmup/final phase handling

4. **Integration Tests:**
   - Full training loop with rank allocation
   - Rank pattern saving after training

## Mathematical Foundation

**AdaLoRA SVD Parameterization:**
```
Delta W = P @ diag(Lambda) @ Q^T
```

**Where:**
- `P` is (out_features, r) - like LoRA B
- `Lambda` is (r,) - singular values (prunable)
- `Q` is (r, in_features) - like LoRA A

**Importance Score:**
```
I(lambda_i) = |lambda_i * grad(lambda_i)|
```

**EMA Update:**
```
S_t = beta * S_{t-1} + (1-beta) * I_t
```

**Budget Schedule:**
```
budget(t) = init_r - (t - tinit) / (total - tfinal - tinit) * (init_r - target_r)
```

## Complexity

- **Implementation:** High
- **Mathematical:** Medium-High
- **Dependencies:** Training loop integration for gradient access

## Priority

**High** - Demonstrates dynamic rank allocation pattern
