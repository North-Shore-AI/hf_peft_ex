defmodule HfPeftEx.Tuners.Adalora.Layer do
  @moduledoc """
  AdaLoRA layer with SVD parameterization and prunable singular values.

  Unlike standard LoRA which uses `W = W0 + B @ A`, AdaLoRA uses an SVD-like
  parameterization with prunable singular values:

      Delta W = P @ diag(Lambda) @ Q^T

  Where:
  - `P` (lora_b) is the left singular vectors: `(out_features, r)`
  - `Lambda` (lora_e) is the singular values: `(r, 1)` - these get pruned
  - `Q` (lora_a) is the right singular vectors: `(r, in_features)`

  The pruning happens by zeroing out singular values based on importance
  scores computed during training.

  ## Example

      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}

      layer = base_layer
      |> Layer.new()
      |> Layer.update_layer("default", 8, 16)

      # Forward pass
      output = Layer.forward(layer, input)

      # Apply pruning mask from RankAllocator
      mask = Nx.tensor([[1], [1], [1], [1], [0], [0], [0], [0]])
      layer = Layer.apply_mask(layer, "default", mask)

  """

  @type t :: %__MODULE__{
          base_layer: map(),
          in_features: pos_integer(),
          out_features: pos_integer(),
          lora_a: %{String.t() => Nx.Tensor.t()},
          lora_b: %{String.t() => Nx.Tensor.t()},
          lora_e: %{String.t() => Nx.Tensor.t()},
          ranknum: %{String.t() => non_neg_integer()},
          scaling: %{String.t() => float()},
          active_adapter: String.t(),
          merged_adapters: [String.t()],
          disable_adapters: boolean(),
          merged: boolean(),
          fan_in_fan_out: boolean()
        }

  defstruct base_layer: nil,
            in_features: 0,
            out_features: 0,
            lora_a: %{},
            lora_b: %{},
            lora_e: %{},
            ranknum: %{},
            scaling: %{},
            active_adapter: "default",
            merged_adapters: [],
            disable_adapters: false,
            merged: false,
            fan_in_fan_out: false

  @doc """
  Creates a new AdaLoRA layer wrapping a base layer.

  ## Arguments

    * `base_layer` - Map with `:weight` and optional `:bias` keys
    * `opts` - Options

  ## Options

    * `:fan_in_fan_out` - If true, weight shape is `{in, out}` (default: `false`)

  ## Examples

      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)

  """
  @spec new(map(), keyword()) :: t()
  def new(base_layer, opts \\ []) do
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)
    {dim0, dim1} = Nx.shape(base_layer.weight)

    {in_features, out_features} =
      if fan_in_fan_out do
        {dim0, dim1}
      else
        {dim1, dim0}
      end

    %__MODULE__{
      base_layer: base_layer,
      in_features: in_features,
      out_features: out_features,
      fan_in_fan_out: fan_in_fan_out
    }
  end

  @doc """
  Adds an adapter with SVD parameterization.

  ## Arguments

    * `layer` - The AdaLoRA layer
    * `adapter_name` - Name for this adapter
    * `r` - Rank (number of singular values)
    * `lora_alpha` - Scaling factor

  ## Examples

      layer = Layer.update_layer(layer, "default", 8, 16)

  """
  @spec update_layer(t(), String.t(), non_neg_integer(), number()) :: t()
  def update_layer(%__MODULE__{} = layer, adapter_name, r, lora_alpha) do
    # Q matrix (like LoRA A): r x in_features
    # Initialize with small random values
    key = Nx.Random.key(System.system_time())

    {lora_a, key} =
      Nx.Random.normal(key, 0.0, 0.02, shape: {r, layer.in_features})

    # P matrix (like LoRA B): out_features x r
    # Initialize to zeros
    lora_b = Nx.broadcast(Nx.tensor(0.0), {layer.out_features, r})

    # Singular values (Lambda): r x 1
    # Initialize to ones (will be pruned during training)
    {lora_e, _key} = {Nx.broadcast(Nx.tensor(1.0), {r, 1}), key}

    # Scaling factor
    scaling = if lora_alpha > 0, do: lora_alpha / r, else: r * 1.0

    layer
    |> put_in([Access.key(:lora_a), adapter_name], lora_a)
    |> put_in([Access.key(:lora_b), adapter_name], lora_b)
    |> put_in([Access.key(:lora_e), adapter_name], lora_e)
    |> put_in([Access.key(:ranknum), adapter_name], r)
    |> put_in([Access.key(:scaling), adapter_name], scaling)
  end

  @doc """
  Forward pass with SVD decomposition.

  Computes:
      output = base_output + x @ (E * A)^T @ B^T * scaling / ranknum

  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{disable_adapters: true} = layer, x) do
    apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)
  end

  def forward(%__MODULE__{merged: true} = layer, x) do
    apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)
  end

  def forward(%__MODULE__{} = layer, x) do
    base_output = apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)

    adapter = layer.active_adapter

    case Map.get(layer.lora_a, adapter) do
      nil ->
        base_output

      lora_a ->
        lora_b = layer.lora_b[adapter]
        lora_e = layer.lora_e[adapter]
        scaling = layer.scaling[adapter]
        ranknum = layer.ranknum[adapter]

        # Compute delta: x @ (E * A)^T @ B^T
        # E is (r, 1), A is (r, in_features)
        # E * A broadcasts E across columns
        e_squeezed = Nx.reshape(lora_e, {Nx.axis_size(lora_e, 0)})
        ea = Nx.multiply(lora_a, Nx.new_axis(e_squeezed, 1))

        # x @ A^T gives (batch, r), then @ B^T gives (batch, out_features)
        delta =
          x
          |> Nx.dot(Nx.transpose(ea))
          |> Nx.dot(Nx.transpose(lora_b))
          |> Nx.multiply(scaling / (ranknum + 1.0e-5))

        Nx.add(base_output, delta)
    end
  end

  @doc """
  Computes the delta weight for merging: B @ diag(E) @ A * scaling / ranknum.
  """
  @spec get_delta_weight(t(), String.t()) :: Nx.Tensor.t()
  def get_delta_weight(%__MODULE__{} = layer, adapter) do
    lora_a = layer.lora_a[adapter]
    lora_b = layer.lora_b[adapter]
    lora_e = layer.lora_e[adapter]
    scaling = layer.scaling[adapter]
    ranknum = layer.ranknum[adapter]

    # Compute B @ diag(E) @ A
    # E is (r, 1), A is (r, in_features)
    e_squeezed = Nx.reshape(lora_e, {Nx.axis_size(lora_e, 0)})
    ea = Nx.multiply(lora_a, Nx.new_axis(e_squeezed, 1))

    # B @ (E * A) = B @ EA
    lora_b
    |> Nx.dot(ea)
    |> Nx.multiply(scaling / (ranknum + 1.0e-5))
  end

  @doc """
  Applies a pruning mask to the singular values.

  The mask should have the same shape as lora_e (r x 1).
  Values of 1.0 keep the singular value, 0.0 prunes it.
  """
  @spec apply_mask(t(), String.t(), Nx.Tensor.t()) :: t()
  def apply_mask(%__MODULE__{} = layer, adapter, mask) do
    new_e = Nx.multiply(layer.lora_e[adapter], mask)
    new_rank = mask |> Nx.sum() |> Nx.to_number() |> trunc()

    layer
    |> put_in([Access.key(:lora_e), adapter], new_e)
    |> put_in([Access.key(:ranknum), adapter], new_rank)
  end

  @doc """
  Merges LoRA weights into the base weight.

  Returns `{updated_layer, new_weight}`.
  """
  @spec merge(t(), String.t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = layer, _adapter, base_weight) do
    {layer, base_weight}
  end

  def merge(%__MODULE__{} = layer, adapter, base_weight) do
    delta = get_delta_weight(layer, adapter)

    delta =
      if layer.fan_in_fan_out do
        Nx.transpose(delta)
      else
        delta
      end

    new_weight = Nx.add(base_weight, delta)
    {%{layer | merged: true, merged_adapters: [adapter | layer.merged_adapters]}, new_weight}
  end

  @doc """
  Unmerges LoRA weights from the base weight.

  Returns `{updated_layer, restored_weight}`.
  """
  @spec unmerge(t(), String.t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = layer, _adapter, base_weight) do
    {layer, base_weight}
  end

  def unmerge(%__MODULE__{} = layer, adapter, base_weight) do
    delta = get_delta_weight(layer, adapter)

    delta =
      if layer.fan_in_fan_out do
        Nx.transpose(delta)
      else
        delta
      end

    new_weight = Nx.subtract(base_weight, delta)
    merged_adapters = List.delete(layer.merged_adapters, adapter)

    {%{layer | merged: false, merged_adapters: merged_adapters}, new_weight}
  end

  # Private helpers

  defp apply_base_layer(base_layer, x, fan_in_fan_out) do
    weight = base_layer.weight

    output =
      if fan_in_fan_out do
        # Weight is (in, out), so x @ W
        Nx.dot(x, weight)
      else
        # Weight is (out, in), so x @ W^T
        Nx.dot(x, Nx.transpose(weight))
      end

    maybe_add_bias(output, base_layer)
  end

  defp maybe_add_bias(output, %{bias: nil}), do: output
  defp maybe_add_bias(output, %{bias: bias}), do: Nx.add(output, bias)
end
