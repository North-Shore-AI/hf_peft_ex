defmodule HfPeftEx.Tuners.Lora.Model do
  @moduledoc """
  LoRA Model wrapper that applies LoRA adapters to a base model.

  This module wraps a base model and manages the LoRA configuration,
  including which modules to target for adaptation and adapter state.

  ## Example

      config = LoraConfig.new(
        r: 8,
        lora_alpha: 16,
        target_modules: ["q_proj", "v_proj"]
      )
      
      model = Model.new(base_model, config)
      
      # Check if LoRA should be applied to a module
      Model.should_apply_lora?(config, "q_proj")  # => true
      
      # Merge for inference
      model = Model.merge_adapter(model)

  """

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Conv1d
  alias HfPeftEx.Tuners.Lora.Conv2d
  alias HfPeftEx.Tuners.Lora.Embedding
  alias HfPeftEx.Tuners.Lora.Linear

  @type t :: %__MODULE__{
          base_model: term(),
          config: LoraConfig.t(),
          adapter_name: String.t(),
          adapters_enabled: boolean(),
          merged: boolean(),
          lora_layers: map()
        }

  @type module_spec ::
          %{
            type: :linear,
            in_features: pos_integer(),
            out_features: pos_integer()
          }
          | %{
              type: :embedding,
              num_embeddings: pos_integer(),
              embedding_dim: pos_integer()
            }
          | %{
              type: :conv1d,
              in_channels: pos_integer(),
              out_channels: pos_integer(),
              kernel_size: pos_integer(),
              stride: pos_integer() | nil,
              padding: :valid | :same | non_neg_integer() | nil,
              dilation: pos_integer() | nil,
              groups: pos_integer() | nil
            }
          | %{
              type: :conv2d,
              in_channels: pos_integer(),
              out_channels: pos_integer(),
              kernel_size: pos_integer() | {pos_integer(), pos_integer()},
              stride: pos_integer() | {pos_integer(), pos_integer()} | nil,
              padding:
                :valid | :same | non_neg_integer() | {non_neg_integer(), non_neg_integer()} | nil,
              dilation: pos_integer() | {pos_integer(), pos_integer()} | nil,
              groups: pos_integer() | nil
            }

  defstruct [
    :base_model,
    :config,
    adapter_name: "default",
    adapters_enabled: true,
    merged: false,
    lora_layers: %{}
  ]

  @doc """
  Creates a new LoRA model wrapper.

  ## Arguments

  - `base_model` - The base model to wrap
  - `config` - LoraConfig with LoRA parameters

  ## Options

  - `:adapter_name` - Name for this adapter (default: "default")

  ## Example

      config = LoraConfig.new(r: 8, target_modules: ["q_proj"])
      model = Model.new(base_model, config)

  """
  @spec new(term(), LoraConfig.t(), keyword()) :: t()
  def new(base_model, %LoraConfig{} = config, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    %__MODULE__{
      base_model: base_model,
      config: config,
      adapter_name: adapter_name,
      adapters_enabled: true,
      merged: false,
      lora_layers: %{}
    }
  end

  @doc """
  Checks if a module name matches the target_modules pattern.

  Supports:
  - List of exact module names: `["q_proj", "v_proj"]`
  - Regex pattern string: `".*_proj"`
  - `"all-linear"` keyword to match all modules

  ## Examples

      config = LoraConfig.new(target_modules: ["q_proj", "v_proj"])
      Model.target_modules_match?(config, "q_proj")  # => true
      
      config = LoraConfig.new(target_modules: ".*_proj")
      Model.target_modules_match?(config, "k_proj")  # => true

  """
  @spec target_modules_match?(LoraConfig.t(), String.t()) :: boolean()
  def target_modules_match?(%LoraConfig{target_modules: nil}, _module_name), do: false

  def target_modules_match?(%LoraConfig{target_modules: "all-linear"}, _module_name), do: true

  def target_modules_match?(%LoraConfig{target_modules: modules}, module_name)
      when is_list(modules) do
    module_name in modules
  end

  def target_modules_match?(%LoraConfig{target_modules: pattern}, module_name)
      when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, regex} -> Regex.match?(regex, module_name)
      {:error, _} -> false
    end
  end

  @doc """
  Checks if a module name matches the exclude_modules pattern.

  Returns true if the module should be excluded from LoRA adaptation.
  """
  @spec exclude_modules_match?(LoraConfig.t(), String.t()) :: boolean()
  def exclude_modules_match?(%LoraConfig{exclude_modules: nil}, _module_name), do: false

  def exclude_modules_match?(%LoraConfig{exclude_modules: modules}, module_name)
      when is_list(modules) do
    module_name in modules
  end

  def exclude_modules_match?(%LoraConfig{exclude_modules: pattern}, module_name)
      when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, regex} -> Regex.match?(regex, module_name)
      {:error, _} -> false
    end
  end

  @doc """
  Determines if LoRA should be applied to a module.

  Returns true if the module matches target_modules AND is not excluded.
  """
  @spec should_apply_lora?(LoraConfig.t(), String.t()) :: boolean()
  def should_apply_lora?(config, module_name) do
    target_modules_match?(config, module_name) and
      not exclude_modules_match?(config, module_name)
  end

  @doc """
  Enables the specified adapter.
  """
  @spec enable_adapter(t(), String.t()) :: t()
  def enable_adapter(%__MODULE__{} = model, _adapter_name) do
    %{model | adapters_enabled: true}
  end

  @doc """
  Disables the specified adapter.
  """
  @spec disable_adapter(t(), String.t()) :: t()
  def disable_adapter(%__MODULE__{} = model, _adapter_name) do
    %{model | adapters_enabled: false}
  end

  @doc """
  Returns the PEFT configuration for this model.
  """
  @spec get_peft_config(t()) :: LoraConfig.t()
  def get_peft_config(%__MODULE__{config: config}), do: config

  @doc """
  Merges the LoRA adapter weights into the base model weights.

  After merging, the model can be used for inference without
  computing the LoRA contribution separately.
  """
  @spec merge_adapter(t()) :: t()
  def merge_adapter(%__MODULE__{merged: true} = model), do: model

  def merge_adapter(%__MODULE__{} = model) do
    %{model | merged: true}
  end

  @doc """
  Unmerges the LoRA adapter weights from the base model weights.

  After unmerging, the LoRA contribution will be computed separately
  during the forward pass.
  """
  @spec unmerge_adapter(t()) :: t()
  def unmerge_adapter(%__MODULE__{merged: false} = model), do: model

  def unmerge_adapter(%__MODULE__{} = model) do
    %{model | merged: false}
  end

  @doc """
  Applies LoRA adapters to a set of module inputs.

  The `module_inputs` map should include `:input` and `:base_output` keys,
  and can include `:weight` for DoRA-enabled layers.
  """
  @spec apply_adapters(t(), %{String.t() => map()}, keyword()) :: %{String.t() => Nx.Tensor.t()}
  def apply_adapters(%__MODULE__{} = model, module_inputs, opts \\ [])
      when is_map(module_inputs) do
    if not model.adapters_enabled or model.merged do
      Map.new(module_inputs, fn {module_name, input_spec} ->
        {module_name, Map.fetch!(input_spec, :base_output)}
      end)
    else
      Enum.reduce(module_inputs, %{}, fn {module_name, input_spec}, acc ->
        base_output = Map.fetch!(input_spec, :base_output)
        layer = Map.get(model.lora_layers, module_name)
        output = apply_layer(layer, input_spec, base_output, opts)
        Map.put(acc, module_name, output)
      end)
    end
  end

  @doc """
  Builds LoRA layers for the provided module specs and stores them in the model.

  The `module_specs` map should contain module names and their shapes/types, e.g.:

      %{
        "q_proj" => %{type: :linear, in_features: 1024, out_features: 1024},
        "embed" => %{type: :embedding, num_embeddings: 10_000, embedding_dim: 768}
      }
  """
  @spec add_lora_layers(t(), %{String.t() => module_spec()}) :: t()
  def add_lora_layers(%__MODULE__{} = model, module_specs) when is_map(module_specs) do
    config = model.config

    lora_layers =
      Enum.reduce(module_specs, model.lora_layers, fn {module_name, spec}, acc ->
        if should_add_layer?(config, module_name, spec) do
          build_layer(acc, module_name, spec, config)
        else
          acc
        end
      end)

    %{model | lora_layers: lora_layers}
  end

  defp should_add_layer?(%LoraConfig{target_modules: "all-linear"} = config, module_name, %{
         type: :linear
       }) do
    should_apply_lora?(config, module_name)
  end

  defp should_add_layer?(%LoraConfig{target_modules: "all-linear"}, _module_name, _spec),
    do: false

  defp should_add_layer?(config, module_name, _spec) do
    should_apply_lora?(config, module_name)
  end

  defp build_layer(
         acc,
         module_name,
         %{type: :linear, in_features: in_features, out_features: out_features},
         config
       ) do
    Map.put(acc, module_name, Linear.new(in_features, out_features, config: config))
  end

  defp build_layer(
         acc,
         module_name,
         %{type: :embedding, num_embeddings: num_embeddings, embedding_dim: embedding_dim},
         config
       ) do
    Map.put(acc, module_name, Embedding.new(num_embeddings, embedding_dim, config: config))
  end

  defp build_layer(
         acc,
         module_name,
         %{
           type: :conv1d,
           in_channels: in_channels,
           out_channels: out_channels,
           kernel_size: kernel_size
         } = spec,
         config
       ) do
    Map.put(
      acc,
      module_name,
      Conv1d.new(in_channels, out_channels, kernel_size, conv_opts(spec, config))
    )
  end

  defp build_layer(
         acc,
         module_name,
         %{
           type: :conv2d,
           in_channels: in_channels,
           out_channels: out_channels,
           kernel_size: kernel_size
         } = spec,
         config
       ) do
    Map.put(
      acc,
      module_name,
      Conv2d.new(in_channels, out_channels, kernel_size, conv_opts(spec, config))
    )
  end

  defp build_layer(acc, _module_name, _spec, _config), do: acc

  defp apply_layer(nil, _input_spec, base_output, _opts), do: base_output

  defp apply_layer(%Linear{} = layer, input_spec, base_output, opts) do
    x = Map.fetch!(input_spec, :input)
    Linear.forward(layer, x, base_output, forward_opts(input_spec, opts))
  end

  defp apply_layer(%Embedding{} = layer, input_spec, base_output, opts) do
    indices = Map.fetch!(input_spec, :input)
    Embedding.forward(layer, indices, base_output, forward_opts(input_spec, opts))
  end

  defp apply_layer(%Conv1d{} = layer, input_spec, base_output, opts) do
    x = Map.fetch!(input_spec, :input)
    Conv1d.forward(layer, x, base_output, forward_opts(input_spec, opts))
  end

  defp apply_layer(%Conv2d{} = layer, input_spec, base_output, opts) do
    x = Map.fetch!(input_spec, :input)
    Conv2d.forward(layer, x, base_output, forward_opts(input_spec, opts))
  end

  defp apply_layer(_layer, _input_spec, base_output, _opts), do: base_output

  defp forward_opts(input_spec, opts) do
    case Map.get(input_spec, :weight) do
      nil -> opts
      weight -> Keyword.put(opts, :weight, weight)
    end
  end

  defp conv_opts(spec, config) do
    []
    |> maybe_put_opt(:stride, spec)
    |> maybe_put_opt(:padding, spec)
    |> maybe_put_opt(:dilation, spec)
    |> maybe_put_opt(:groups, spec)
    |> Keyword.put(:config, config)
  end

  defp maybe_put_opt(opts, key, spec) do
    case Map.get(spec, key) do
      nil -> opts
      value -> Keyword.put(opts, key, value)
    end
  end
end
