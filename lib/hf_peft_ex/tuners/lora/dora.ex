defmodule HfPeftEx.Tuners.Lora.Dora do
  @moduledoc """
  DoRA (Weight-Decomposed LoRA) helper functions for linear layers.

  This module mirrors the core math used in Hugging Face's PEFT DoRA
  implementation for linear layers. It computes a per-output magnitude
  vector and applies the DoRA scaling to the LoRA contribution.
  """

  @type t :: %__MODULE__{
          magnitude: Nx.Tensor.t(),
          fan_in_fan_out: boolean()
        }

  defstruct [:magnitude, fan_in_fan_out: false]

  @doc """
  Initializes the DoRA magnitude vector from the base weight.

  Computes row-wise L2 norms of the weight (per output feature).
  """
  @spec init_magnitude(Nx.Tensor.t()) :: Nx.Tensor.t()
  def init_magnitude(weight) do
    weight
    |> Nx.multiply(weight)
    |> Nx.sum(axes: [1])
    |> Nx.sqrt()
  end

  @doc """
  Creates a DoRA struct from the base weight.

  ## Options

  - `:fan_in_fan_out` - Whether the weight is stored as {in, out}. Default: false.
  """
  @spec new(Nx.Tensor.t(), keyword()) :: t()
  def new(weight, opts \\ []) do
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)
    weight_for_norm = maybe_transpose(weight, fan_in_fan_out)

    %__MODULE__{
      magnitude: init_magnitude(weight_for_norm),
      fan_in_fan_out: fan_in_fan_out
    }
  end

  @doc """
  Computes the row-wise L2 norm of the combined base weight and LoRA weight.

  The LoRA weight is scaled by `scaling` before the norm is computed.
  """
  @spec get_weight_norm(Nx.Tensor.t(), Nx.Tensor.t(), number(), keyword()) :: Nx.Tensor.t()
  def get_weight_norm(weight, lora_weight, scaling, opts \\ []) do
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)

    weight
    |> maybe_transpose(fan_in_fan_out)
    |> Nx.add(Nx.multiply(lora_weight, scaling))
    |> init_magnitude()
  end

  @doc """
  Computes the DoRA delta output for a linear layer.

  The result should be added to the base output.
  """
  @spec forward(
          t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          number(),
          Nx.Tensor.t(),
          Nx.Tensor.t() | nil
        ) ::
          Nx.Tensor.t()
  def forward(%__MODULE__{} = dora, x, lora_a, lora_b, scaling, weight, base_result \\ nil) do
    lora_weight = Nx.dot(lora_b, lora_a)

    weight_norm =
      get_weight_norm(weight, lora_weight, scaling, fan_in_fan_out: dora.fan_in_fan_out)

    mag_norm_scale = Nx.divide(dora.magnitude, weight_norm)
    {out_features} = Nx.shape(mag_norm_scale)
    mag_norm_scale = Nx.reshape(mag_norm_scale, {1, out_features})

    lora_result =
      x
      |> Nx.dot(Nx.transpose(lora_a))
      |> Nx.dot(Nx.transpose(lora_b))

    base_result =
      case base_result do
        nil ->
          weight_for_linear = maybe_transpose(weight, dora.fan_in_fan_out)
          Nx.dot(x, Nx.transpose(weight_for_linear))

        provided ->
          provided
      end

    mag_norm_scale
    |> Nx.subtract(1.0)
    |> Nx.multiply(base_result)
    |> Nx.add(Nx.multiply(mag_norm_scale, Nx.multiply(lora_result, scaling)))
  end

  @doc """
  Applies DoRA to the provided layer map and adds it to the base output.
  """
  @spec apply_dora(
          t(),
          %{lora_A: Nx.Tensor.t(), lora_B: Nx.Tensor.t(), scaling: number()},
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: Nx.Tensor.t()
  def apply_dora(
        %__MODULE__{} = dora,
        %{lora_A: lora_a, lora_B: lora_b, scaling: scaling},
        x,
        base_output,
        weight
      ) do
    dora_delta = forward(dora, x, lora_a, lora_b, scaling, weight, base_output)
    Nx.add(base_output, dora_delta)
  end

  defp maybe_transpose(weight, true), do: Nx.transpose(weight)
  defp maybe_transpose(weight, false), do: weight
end
