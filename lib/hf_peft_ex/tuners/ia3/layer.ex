defmodule HfPeftEx.Tuners.IA3.Layer do
  @moduledoc """
  IA3 layer implementation with multiplicative scaling vectors.

  IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) learns
  a scaling vector that is applied element-wise to layer activations.

  ## Forward Pass

  For non-feedforward layers:

      y = (Wx + b) * l

  For feedforward layers:

      y = W(x * l) + b = (Wx) * l + b

  Where `l` is the learned scaling vector.

  ## Merge/Unmerge

  When merged, the base weights are modified:

      W' = W * diag(l)  (for rows or columns depending on feedforward)

  This allows the model to run without the adapter overhead at inference time.

  ## Example

      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer, is_feedforward: false)
      |> Layer.update_layer("default", true)

      output = Layer.forward(layer, input)

  """

  @type t :: %__MODULE__{
          base_layer: map(),
          in_features: pos_integer(),
          out_features: pos_integer(),
          is_feedforward: boolean(),
          fan_in_fan_out: boolean(),
          ia3_l: %{String.t() => Nx.Tensor.t()},
          active_adapter: String.t(),
          merged_adapters: [String.t()],
          disable_adapters: boolean(),
          merged: boolean()
        }

  defstruct [
    :base_layer,
    :in_features,
    :out_features,
    :is_feedforward,
    fan_in_fan_out: false,
    ia3_l: %{},
    active_adapter: "default",
    merged_adapters: [],
    disable_adapters: false,
    merged: false
  ]

  @doc """
  Creates a new IA3 layer wrapping a base layer.

  ## Arguments

  - `base_layer` - Map containing `:weight` (and optionally `:bias`)
  - `is_feedforward` - Whether this is a feedforward module

  ## Options

  - `:fan_in_fan_out` - Set `true` if weight is stored as `{in_features, out_features}`

  ## Examples

      base = %{weight: Nx.random_normal({64, 32})}
      layer = Layer.new(base, false)

      # With fan_in_fan_out
      layer = Layer.new(base, false, fan_in_fan_out: true)

  """
  @spec new(map(), boolean(), keyword()) :: t()
  def new(base_layer, is_feedforward, opts \\ []) do
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)
    {in_features, out_features} = get_dimensions(base_layer.weight, fan_in_fan_out)

    %__MODULE__{
      base_layer: base_layer,
      in_features: in_features,
      out_features: out_features,
      is_feedforward: is_feedforward,
      fan_in_fan_out: fan_in_fan_out,
      ia3_l: %{},
      active_adapter: "default",
      merged_adapters: [],
      disable_adapters: false,
      merged: false
    }
  end

  defp get_dimensions(weight, fan_in_fan_out) do
    {dim0, dim1} = {Nx.axis_size(weight, 0), Nx.axis_size(weight, 1)}

    if fan_in_fan_out do
      # Weight stored as {in_features, out_features}
      {dim0, dim1}
    else
      # Weight stored as {out_features, in_features}
      {dim1, dim0}
    end
  end

  @doc """
  Adds an adapter to this layer.

  ## Arguments

  - `layer` - The IA3 layer
  - `adapter_name` - Name for the adapter
  - `init_ia3_weights` - If `true`, initialize to ones (identity). If `false`, random init.

  ## Examples

      layer = Layer.update_layer(layer, "default", true)
      layer = Layer.update_layer(layer, "task_specific", false)

  """
  @spec update_layer(t(), String.t(), boolean()) :: t()
  def update_layer(%__MODULE__{} = layer, adapter_name, init_ia3_weights) do
    dim =
      if layer.is_feedforward do
        layer.in_features
      else
        layer.out_features
      end

    ia3_l =
      if init_ia3_weights do
        Nx.broadcast(Nx.tensor(1.0), {dim})
      else
        # Small random initialization
        key = Nx.Random.key(System.system_time())
        {tensor, _key} = Nx.Random.normal(key, 0.0, 0.02, shape: {dim})
        tensor
      end

    put_in(layer.ia3_l[adapter_name], ia3_l)
  end

  @doc """
  Resets the IA3 parameters for an adapter to ones.
  """
  @spec reset_ia3_parameters(t(), String.t()) :: t()
  def reset_ia3_parameters(%__MODULE__{} = layer, adapter_name) do
    case Map.get(layer.ia3_l, adapter_name) do
      nil ->
        layer

      ia3_l ->
        {dim} = Nx.shape(ia3_l)
        put_in(layer.ia3_l[adapter_name], Nx.broadcast(Nx.tensor(1.0), {dim}))
    end
  end

  @doc """
  Sets the active adapter for this layer.
  """
  @spec set_adapter(t(), String.t()) :: t()
  def set_adapter(%__MODULE__{} = layer, adapter_name) do
    %{layer | active_adapter: adapter_name}
  end

  @doc """
  Deletes an adapter from this layer.
  """
  @spec delete_adapter(t(), String.t()) :: t()
  def delete_adapter(%__MODULE__{} = layer, adapter_name) do
    %{layer | ia3_l: Map.delete(layer.ia3_l, adapter_name)}
  end

  @doc """
  Forward pass with IA3 scaling.

  When adapters are disabled or the layer is merged, returns the base layer output.
  Otherwise, applies the IA3 scaling vector to the output.
  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{disable_adapters: true} = layer, x) do
    apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)
  end

  def forward(%__MODULE__{merged: true} = layer, x) do
    apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)
  end

  def forward(%__MODULE__{} = layer, x) do
    ia3_l = Map.fetch!(layer.ia3_l, layer.active_adapter)

    # Compute base layer output
    base_output = apply_base_layer(layer.base_layer, x, layer.fan_in_fan_out)

    # Apply IA3 scaling (works the same for both feedforward and non-feedforward
    # because for linear layers, scaling input vs output is mathematically equivalent)
    apply_ia3_scaling(base_output, ia3_l, layer.base_layer)
  end

  defp apply_base_layer(base_layer, x, fan_in_fan_out) do
    weight = base_layer.weight
    weight = if fan_in_fan_out, do: Nx.transpose(weight), else: weight
    output = Nx.dot(x, Nx.transpose(weight))
    maybe_add_bias(output, base_layer)
  end

  defp apply_ia3_scaling(output, ia3_l, _base_layer) do
    # Scale the output (before adding bias for non-feedforward)
    # Note: In PyTorch PEFT, bias is scaled too for non-feedforward.
    # We apply scaling to the full output including bias.
    Nx.multiply(output, ia3_l)
  end

  defp maybe_add_bias(output, %{bias: nil}), do: output
  defp maybe_add_bias(output, %{bias: bias}), do: Nx.add(output, bias)

  @doc """
  Merges IA3 scaling into base layer weights.

  After merging, the forward pass will use the modified weights directly
  without computing the IA3 contribution separately.

  Returns `{:ok, merged_layer}` or `{:error, :already_merged}`.
  """
  @spec merge(t(), keyword()) :: {:ok, t()} | {:error, :already_merged}
  def merge(layer, opts \\ [])

  def merge(%__MODULE__{merged: true}, _opts) do
    {:error, :already_merged}
  end

  def merge(%__MODULE__{} = layer, opts) do
    adapter_name = Keyword.get(opts, :adapter_name, layer.active_adapter)
    ia3_l = Map.fetch!(layer.ia3_l, adapter_name)

    base_weight = layer.base_layer.weight
    base_bias = Map.get(layer.base_layer, :bias)

    {new_weight, new_bias} =
      if layer.is_feedforward do
        # Scale columns: W[:, i] *= ia3_l[i]
        scaling = Nx.reshape(ia3_l, {1, :auto})
        new_weight = Nx.multiply(base_weight, scaling)
        {new_weight, base_bias}
      else
        # Scale rows: W[i, :] *= ia3_l[i]
        scaling = Nx.reshape(ia3_l, {:auto, 1})
        new_weight = Nx.multiply(base_weight, scaling)
        # Also scale bias
        new_bias =
          if base_bias do
            Nx.multiply(base_bias, ia3_l)
          else
            nil
          end

        {new_weight, new_bias}
      end

    new_base_layer = %{layer.base_layer | weight: new_weight}

    new_base_layer =
      if base_bias != nil do
        %{new_base_layer | bias: new_bias}
      else
        new_base_layer
      end

    {:ok,
     %{
       layer
       | base_layer: new_base_layer,
         merged: true,
         merged_adapters: [adapter_name | layer.merged_adapters]
     }}
  end

  @doc """
  Unmerges IA3 scaling from base layer weights.

  Restores the original weights by dividing by the scaling vectors.

  Returns `{:ok, unmerged_layer}` or `{:error, :not_merged}`.
  """
  @spec unmerge(t()) :: {:ok, t()} | {:error, :not_merged}
  def unmerge(%__MODULE__{merged: false}) do
    {:error, :not_merged}
  end

  def unmerge(%__MODULE__{} = layer) do
    # Unmerge by reversing through merged adapters
    new_base_layer =
      Enum.reduce(Enum.reverse(layer.merged_adapters), layer.base_layer, fn adapter_name, bl ->
        unmerge_adapter(layer, bl, adapter_name)
      end)

    {:ok,
     %{
       layer
       | base_layer: new_base_layer,
         merged: false,
         merged_adapters: []
     }}
  end

  defp unmerge_adapter(layer, bl, adapter_name) do
    ia3_l = Map.fetch!(layer.ia3_l, adapter_name)
    # Add small epsilon to avoid division by zero
    ia3_l_safe = Nx.add(ia3_l, 1.0e-8)

    base_weight = bl.weight
    base_bias = Map.get(bl, :bias)

    {new_weight, new_bias} =
      compute_unmerged_weights(layer.is_feedforward, base_weight, base_bias, ia3_l_safe)

    updated_layer = %{bl | weight: new_weight}
    maybe_update_bias(updated_layer, base_bias, new_bias)
  end

  defp compute_unmerged_weights(true, base_weight, base_bias, ia3_l_safe) do
    # Feedforward: divide columns
    scaling = Nx.reshape(ia3_l_safe, {1, :auto})
    new_weight = Nx.divide(base_weight, scaling)
    {new_weight, base_bias}
  end

  defp compute_unmerged_weights(false, base_weight, base_bias, ia3_l_safe) do
    # Non-feedforward: divide rows
    scaling = Nx.reshape(ia3_l_safe, {:auto, 1})
    new_weight = Nx.divide(base_weight, scaling)
    # Also restore bias
    new_bias = if base_bias, do: Nx.divide(base_bias, ia3_l_safe), else: nil
    {new_weight, new_bias}
  end

  defp maybe_update_bias(updated_layer, nil, _new_bias), do: updated_layer

  defp maybe_update_bias(updated_layer, _base_bias, new_bias),
    do: %{updated_layer | bias: new_bias}
end
