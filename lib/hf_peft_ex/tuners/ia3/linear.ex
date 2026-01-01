defmodule HfPeftEx.Tuners.IA3.Linear do
  @moduledoc """
  IA3 wrapper for Linear (dense) layers.

  This module provides IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
  for dense/linear layers, supporting the `fan_in_fan_out` transposition mode where weights
  are stored as `{in_features, out_features}` instead of `{out_features, in_features}`.

  Unlike LoRA which learns low-rank matrices, IA3 learns a single scaling vector that is
  applied element-wise to layer outputs.

  ## Example

      config = IA3Config.new(init_ia3_weights: true)
      linear = Linear.new(1024, 512, config: config)

      # Forward pass with base layer output
      result = Linear.forward(linear, input, base_output)

      # Merge for inference
      {merged, new_weight} = Linear.merge(linear, base_weight)

  ## IA3 Math

  For non-feedforward layers:

      output = base_output * ia3_l

  For feedforward layers (scaling applied to input dimension):

      output = base_output * ia3_l

  The difference is in the dimension of `ia3_l`:
  - Non-feedforward: `{out_features}`
  - Feedforward: `{in_features}`

  """

  alias HfPeftEx.Tuners.IA3.Config, as: IA3Config

  @type t :: %__MODULE__{
          in_features: pos_integer(),
          out_features: pos_integer(),
          ia3_l: Nx.Tensor.t(),
          merged: boolean(),
          is_feedforward: boolean(),
          fan_in_fan_out: boolean()
        }

  defstruct [
    :in_features,
    :out_features,
    :ia3_l,
    merged: false,
    is_feedforward: false,
    fan_in_fan_out: false
  ]

  @doc """
  Creates a new Linear IA3 layer.

  ## Arguments

    * `in_features` - Input dimension
    * `out_features` - Output dimension
    * `opts` - Options

  ## Options

    * `:init_ia3_weights` - Initialize to ones (default: true)
    * `:is_feedforward` - Whether this is a feedforward layer (default: false)
    * `:fan_in_fan_out` - Weight stored as {in, out} (default: false)
    * `:config` - IA3Config struct to extract parameters from

  ## Examples

      linear = Linear.new(1024, 512)

      config = IA3Config.new(init_ia3_weights: false)
      linear = Linear.new(1024, 512, config: config)

  """
  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  def new(in_features, out_features, opts \\ []) do
    opts = apply_config(opts)

    init_ia3_weights = Keyword.get(opts, :init_ia3_weights, true)
    is_feedforward = Keyword.get(opts, :is_feedforward, false)
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)

    dim =
      if is_feedforward do
        in_features
      else
        out_features
      end

    ia3_l = init_ia3_l(dim, init_ia3_weights)

    %__MODULE__{
      in_features: in_features,
      out_features: out_features,
      ia3_l: ia3_l,
      merged: false,
      is_feedforward: is_feedforward,
      fan_in_fan_out: fan_in_fan_out
    }
  end

  defp apply_config(opts) do
    case Keyword.get(opts, :config) do
      nil ->
        opts

      %IA3Config{} = config ->
        opts
        |> Keyword.put_new(:init_ia3_weights, config.init_ia3_weights)
        |> Keyword.put_new(:fan_in_fan_out, config.fan_in_fan_out)
    end
  end

  defp init_ia3_l(dim, true) do
    Nx.broadcast(Nx.tensor(1.0), {dim})
  end

  defp init_ia3_l(dim, false) do
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.normal(key, 0.0, 0.02, shape: {dim})
    tensor
  end

  @doc """
  Forward pass through the Linear IA3 layer.

  When merged, returns the base output unchanged.
  When not merged, scales the base output by `ia3_l`.
  """
  @spec forward(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{merged: true}, _x, base_output) do
    base_output
  end

  def forward(%__MODULE__{} = linear, _x, base_output) do
    Nx.multiply(base_output, linear.ia3_l)
  end

  @doc """
  Returns the IA3 scaling vector.
  """
  @spec get_scaling_vector(t()) :: Nx.Tensor.t()
  def get_scaling_vector(%__MODULE__{ia3_l: ia3_l}) do
    ia3_l
  end

  @doc """
  Merges the IA3 scaling into the base weight.

  Returns `{updated_linear, new_weight}` where the linear is marked as merged
  and the weight is scaled by `ia3_l`.

  For non-feedforward layers: rows are scaled
  For feedforward layers: columns are scaled
  When `fan_in_fan_out` is true, the dimension interpretation is swapped.
  """
  @spec merge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = linear, base_weight) do
    {linear, base_weight}
  end

  def merge(%__MODULE__{} = linear, base_weight) do
    new_weight = apply_scaling_to_weight(linear, base_weight, :multiply)
    {%{linear | merged: true}, new_weight}
  end

  @doc """
  Unmerges the IA3 scaling from the base weight.

  Returns `{updated_linear, new_weight}` where the linear is marked as unmerged
  and the weight is divided by `ia3_l`.
  """
  @spec unmerge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = linear, base_weight) do
    {linear, base_weight}
  end

  def unmerge(%__MODULE__{} = linear, base_weight) do
    new_weight = apply_scaling_to_weight(linear, base_weight, :divide)
    {%{linear | merged: false}, new_weight}
  end

  defp apply_scaling_to_weight(linear, base_weight, operation) do
    ia3_l = linear.ia3_l
    # Add epsilon for division
    ia3_l_safe = if operation == :divide, do: Nx.add(ia3_l, 1.0e-8), else: ia3_l

    # Determine if we scale rows or columns based on is_feedforward and fan_in_fan_out
    scale_columns? = determine_scale_direction(linear)

    if scale_columns? do
      # Scale columns: W[:, i] *= ia3_l[i] (or divide)
      scaling = Nx.reshape(ia3_l_safe, {1, :auto})
      apply_op(base_weight, scaling, operation)
    else
      # Scale rows: W[i, :] *= ia3_l[i] (or divide)
      scaling = Nx.reshape(ia3_l_safe, {:auto, 1})
      apply_op(base_weight, scaling, operation)
    end
  end

  defp determine_scale_direction(%{is_feedforward: is_ff, fan_in_fan_out: fif}) do
    # For non-feedforward: scale rows (out_features dimension)
    # For feedforward: scale columns (in_features dimension)
    # fan_in_fan_out swaps the interpretation
    case {is_ff, fif} do
      # scale rows
      {false, false} -> false
      # fan_in_fan_out: scale columns
      {false, true} -> true
      # feedforward: scale columns
      {true, false} -> true
      # both: scale rows
      {true, true} -> false
    end
  end

  defp apply_op(weight, scaling, :multiply), do: Nx.multiply(weight, scaling)
  defp apply_op(weight, scaling, :divide), do: Nx.divide(weight, scaling)

  @doc """
  Resets the IA3 scaling vector to ones.
  """
  @spec reset_ia3_parameters(t()) :: t()
  def reset_ia3_parameters(%__MODULE__{} = linear) do
    {dim} = Nx.shape(linear.ia3_l)
    %{linear | ia3_l: Nx.broadcast(Nx.tensor(1.0), {dim})}
  end

  @doc """
  Returns the number of trainable parameters.

  IA3 only adds `d` parameters where `d` is the scaling vector dimension.
  """
  @spec trainable_params(t()) :: pos_integer()
  def trainable_params(%__MODULE__{ia3_l: ia3_l}) do
    {dim} = Nx.shape(ia3_l)
    dim
  end
end
