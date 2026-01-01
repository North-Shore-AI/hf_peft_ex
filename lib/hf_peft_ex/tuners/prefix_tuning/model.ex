defmodule HfPeftEx.Tuners.PrefixTuning.Model do
  @moduledoc """
  Prefix Tuning Model wrapper.

  This module wraps a base model with prefix tuning capabilities.
  It manages the prefix encoder and provides methods for generating
  past_key_values that are prepended to attention layers.

  ## Usage

      base_model = load_your_model()

      config = HfPeftEx.Tuners.PrefixTuning.Config.new(
        num_virtual_tokens: 20,
        num_layers: 12,
        token_dim: 768,
        num_attention_heads: 12,
        prefix_projection: true,
        encoder_hidden_size: 512
      )

      model = HfPeftEx.Tuners.PrefixTuning.Model.new(base_model, config)

      # Get past_key_values for attention layers
      past_key_values = Model.get_past_key_values(model, batch_size)

      # Extend attention mask for prefix
      attention_mask = Model.prepare_attention_mask(model, batch_size, original_mask)

  ## Integration with Transformers

  The `get_past_key_values/2` function returns a list of `{key, value}` tuples,
  one per transformer layer. These should be concatenated with the input's
  key and value tensors in each attention layer:

      # For each layer l:
      K_l = concat([prefix_key_l, input_key_l], axis=seq_dim)
      V_l = concat([prefix_value_l, input_value_l], axis=seq_dim)

  """

  alias HfPeftEx.Tuners.PrefixTuning.{Config, Encoder}

  @type t :: %__MODULE__{
          config: Config.t(),
          base_model: map(),
          prefix_encoder: Encoder.t()
        }

  defstruct [:config, :base_model, :prefix_encoder]

  @doc """
  Creates a new Prefix Tuning model wrapper.

  The model will extract configuration from the base model to fill in
  missing values (num_layers, num_attention_heads, token_dim).

  ## Parameters

  - `base_model` - The base model to wrap (expects a map with `:config` key)
  - `config` - The PrefixTuning configuration

  ## Examples

      model = Model.new(base_model, config)

  """
  @spec new(map(), Config.t()) :: t()
  def new(base_model, %Config{} = config) do
    # Auto-fill config from base model if not provided
    config = fill_config_from_model(config, base_model)

    # Create the prefix encoder
    prefix_encoder = Encoder.new(config)

    %__MODULE__{
      config: config,
      base_model: base_model,
      prefix_encoder: prefix_encoder
    }
  end

  defp fill_config_from_model(%Config{} = config, base_model) do
    base_config = Map.get(base_model, :config, %{})

    config
    |> maybe_fill(:num_layers, base_config, :num_hidden_layers)
    |> maybe_fill(:num_attention_heads, base_config, :num_attention_heads)
    |> maybe_fill(:token_dim, base_config, :hidden_size)
  end

  defp maybe_fill(config, config_key, base_config, base_key) do
    if Map.get(config, config_key) == nil do
      value = Map.get(base_config, base_key)
      if value, do: Map.put(config, config_key, value), else: config
    else
      config
    end
  end

  @doc """
  Generates past_key_values for prepending to attention layers.

  Returns a list of `{key, value}` tuples, one per transformer layer.
  Each tensor has shape `(batch_size, num_heads, num_virtual_tokens, head_dim)`.

  ## Examples

      past_key_values = Model.get_past_key_values(model, 4)
      # => [{key_0, value_0}, {key_1, value_1}, ...]

  """
  @spec get_past_key_values(t(), pos_integer()) :: [{Nx.Tensor.t(), Nx.Tensor.t()}]
  def get_past_key_values(%__MODULE__{} = model, batch_size) do
    # Get full tensor: (batch, num_layers, 2, num_heads, num_tokens, head_dim)
    full_tensor = Encoder.forward(model.prefix_encoder, batch_size)

    # Split into list of per-layer (key, value) tuples
    num_layers = model.config.num_layers

    for layer_idx <- 0..(num_layers - 1) do
      # Extract layer slice: (batch, 2, num_heads, num_tokens, head_dim)
      layer_kv = full_tensor[[.., layer_idx, .., .., .., ..]]

      # Split into key and value: each (batch, num_heads, num_tokens, head_dim)
      key = layer_kv[[.., 0, .., .., ..]]
      value = layer_kv[[.., 1, .., .., ..]]

      {key, value}
    end
  end

  @doc """
  Returns the full past_key_values tensor.

  Shape: `(batch_size, num_layers, 2, num_heads, num_virtual_tokens, head_dim)`

  This is useful when you need the raw tensor rather than the list format.

  ## Examples

      tensor = Model.get_past_key_values_tensor(model, 4)

  """
  @spec get_past_key_values_tensor(t(), pos_integer()) :: Nx.Tensor.t()
  def get_past_key_values_tensor(%__MODULE__{} = model, batch_size) do
    Encoder.forward(model.prefix_encoder, batch_size)
  end

  @doc """
  Prepares an attention mask for prefix tuning.

  Prepends ones for the prefix token positions to the attention mask.
  If no attention mask is provided, creates one for just the prefix tokens.

  ## Parameters

  - `model` - The prefix tuning model
  - `batch_size` - Batch size for the mask
  - `attention_mask` - Optional existing attention mask with shape `(batch_size, seq_len)`

  ## Returns

  A tensor with shape `(batch_size, num_virtual_tokens + seq_len)` or
  `(batch_size, num_virtual_tokens)` if no mask was provided.

  ## Examples

      # Extend existing mask
      new_mask = Model.prepare_attention_mask(model, 4, original_mask)

      # Create prefix-only mask
      prefix_mask = Model.prepare_attention_mask(model, 4, nil)

  """
  @spec prepare_attention_mask(t(), pos_integer(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def prepare_attention_mask(%__MODULE__{} = model, batch_size, nil) do
    # Create mask of all ones for just the prefix tokens
    Nx.broadcast(1, {batch_size, model.config.num_virtual_tokens})
    |> Nx.as_type(:s64)
  end

  def prepare_attention_mask(%__MODULE__{} = model, batch_size, attention_mask) do
    # Create prefix mask of all ones
    prefix_mask =
      Nx.broadcast(1, {batch_size, model.config.num_virtual_tokens})
      |> Nx.as_type(Nx.type(attention_mask))

    # Concatenate: [prefix_mask, original_mask]
    Nx.concatenate([prefix_mask, attention_mask], axis: 1)
  end

  @doc """
  Returns a map of all trainable parameters.

  The parameter names are prefixed with "prefix_encoder." to indicate
  they belong to the prefix encoder component.

  ## Examples

      params = Model.get_trainable_params(model)
      # => %{"prefix_encoder.embedding" => tensor, ...}

  """
  @spec get_trainable_params(t()) :: %{String.t() => Nx.Tensor.t()}
  def get_trainable_params(%__MODULE__{} = model) do
    model.prefix_encoder
    |> Encoder.get_trainable_params()
    |> Map.new(fn {name, tensor} -> {"prefix_encoder.#{name}", tensor} end)
  end

  @doc """
  Updates the model with new parameter values.

  Parameter names should be prefixed with "prefix_encoder." as returned
  by `get_trainable_params/1`.

  ## Examples

      new_params = %{"prefix_encoder.embedding" => updated_embedding}
      updated_model = Model.set_trainable_params(model, new_params)

  """
  @spec set_trainable_params(t(), %{String.t() => Nx.Tensor.t()}) :: t()
  def set_trainable_params(%__MODULE__{} = model, params) do
    # Strip the "prefix_encoder." prefix
    encoder_params =
      params
      |> Enum.filter(fn {k, _v} -> String.starts_with?(k, "prefix_encoder.") end)
      |> Map.new(fn {k, v} ->
        name = String.replace_prefix(k, "prefix_encoder.", "")
        {name, v}
      end)

    updated_encoder = Encoder.set_trainable_params(model.prefix_encoder, encoder_params)
    %{model | prefix_encoder: updated_encoder}
  end

  @doc """
  Returns the total number of trainable parameters.

  ## Examples

      count = Model.param_count(model)

  """
  @spec param_count(t()) :: non_neg_integer()
  def param_count(%__MODULE__{} = model) do
    Encoder.param_count(model.prefix_encoder)
  end

  @doc """
  Returns a summary of trainable parameters.

  ## Returns

  A map with:
  - `:trainable_params` - Total number of trainable parameters
  - `:prefix_encoder_params` - Parameters in the prefix encoder
  - `:params_by_layer` - Breakdown by parameter name

  ## Examples

      summary = Model.trainable_summary(model)

  """
  @spec trainable_summary(t()) :: %{
          trainable_params: non_neg_integer(),
          prefix_encoder_params: non_neg_integer(),
          params_by_layer: %{String.t() => non_neg_integer()}
        }
  def trainable_summary(%__MODULE__{} = model) do
    params = get_trainable_params(model)

    params_by_layer =
      Map.new(params, fn {name, tensor} ->
        {name, Nx.size(tensor)}
      end)

    total = Enum.reduce(params_by_layer, 0, fn {_k, v}, acc -> acc + v end)

    %{
      trainable_params: total,
      prefix_encoder_params: total,
      params_by_layer: params_by_layer
    }
  end
end
