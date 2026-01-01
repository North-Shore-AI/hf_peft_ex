defmodule HfPeftEx.Tuners.IA3.Model do
  @moduledoc """
  IA3 Model wrapper that applies IA3 adapters to a base model.

  This module wraps a base model and manages the IA3 configuration,
  including which modules to target for adaptation and adapter state.

  ## Example

      config = IA3Config.new(
        target_modules: ["q_proj", "v_proj", "down_proj"],
        feedforward_modules: ["down_proj"]
      )

      model = Model.new(base_model, config)

      # Check if IA3 should be applied to a module
      Model.should_apply_ia3?(config, "q_proj")  # => true

      # Merge for inference
      model = Model.merge_adapter(model)

  """

  alias HfPeftEx.Tuners.IA3.Config, as: IA3Config
  alias HfPeftEx.Tuners.IA3.Linear

  @type t :: %__MODULE__{
          base_model: term(),
          config: IA3Config.t(),
          adapter_name: String.t(),
          adapters_enabled: boolean(),
          merged: boolean(),
          ia3_layers: map()
        }

  @type module_spec ::
          %{
            type: :linear,
            in_features: pos_integer(),
            out_features: pos_integer()
          }

  defstruct [
    :base_model,
    :config,
    adapter_name: "default",
    adapters_enabled: true,
    merged: false,
    ia3_layers: %{}
  ]

  @doc """
  Creates a new IA3 model wrapper.

  ## Arguments

  - `base_model` - The base model to wrap
  - `config` - IA3Config with IA3 parameters

  ## Options

  - `:adapter_name` - Name for this adapter (default: "default")

  ## Example

      config = IA3Config.new(target_modules: ["q_proj"])
      model = Model.new(base_model, config)

  """
  @spec new(term(), IA3Config.t(), keyword()) :: t()
  def new(base_model, %IA3Config{} = config, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    %__MODULE__{
      base_model: base_model,
      config: config,
      adapter_name: adapter_name,
      adapters_enabled: true,
      merged: false,
      ia3_layers: %{}
    }
  end

  @doc """
  Checks if a module name matches the target_modules pattern.

  Supports:
  - List of exact module names: `["q_proj", "v_proj"]`
  - Regex pattern string: `".*_proj"`

  ## Examples

      config = IA3Config.new(target_modules: ["q_proj", "v_proj"])
      Model.target_modules_match?(config, "q_proj")  # => true

      config = IA3Config.new(target_modules: ".*_proj")
      Model.target_modules_match?(config, "k_proj")  # => true

  """
  @spec target_modules_match?(IA3Config.t(), String.t()) :: boolean()
  def target_modules_match?(%IA3Config{target_modules: nil}, _module_name), do: false

  def target_modules_match?(%IA3Config{target_modules: modules}, module_name)
      when is_list(modules) do
    module_name in modules
  end

  def target_modules_match?(%IA3Config{target_modules: pattern}, module_name)
      when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, regex} -> Regex.match?(regex, module_name)
      {:error, _} -> false
    end
  end

  @doc """
  Checks if a module name matches the feedforward_modules pattern.
  """
  @spec feedforward_modules_match?(IA3Config.t(), String.t()) :: boolean()
  def feedforward_modules_match?(%IA3Config{feedforward_modules: nil}, _module_name), do: false

  def feedforward_modules_match?(%IA3Config{feedforward_modules: modules}, module_name)
      when is_list(modules) do
    module_name in modules
  end

  def feedforward_modules_match?(%IA3Config{feedforward_modules: pattern}, module_name)
      when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, regex} -> Regex.match?(regex, module_name)
      {:error, _} -> false
    end
  end

  @doc """
  Checks if a module name matches the exclude_modules pattern.

  Returns true if the module should be excluded from IA3 adaptation.
  """
  @spec exclude_modules_match?(IA3Config.t(), String.t()) :: boolean()
  def exclude_modules_match?(%IA3Config{exclude_modules: nil}, _module_name), do: false

  def exclude_modules_match?(%IA3Config{exclude_modules: modules}, module_name)
      when is_list(modules) do
    module_name in modules
  end

  def exclude_modules_match?(%IA3Config{exclude_modules: pattern}, module_name)
      when is_binary(pattern) do
    case Regex.compile(pattern) do
      {:ok, regex} -> Regex.match?(regex, module_name)
      {:error, _} -> false
    end
  end

  @doc """
  Determines if IA3 should be applied to a module.

  Returns true if the module matches target_modules AND is not excluded.
  """
  @spec should_apply_ia3?(IA3Config.t(), String.t()) :: boolean()
  def should_apply_ia3?(config, module_name) do
    target_modules_match?(config, module_name) and
      not exclude_modules_match?(config, module_name)
  end

  @doc """
  Builds IA3 layers for the provided module specs and stores them in the model.

  The `module_specs` map should contain module names and their shapes/types, e.g.:

      %{
        "q_proj" => %{type: :linear, in_features: 1024, out_features: 1024},
        "down_proj" => %{type: :linear, in_features: 4096, out_features: 1024}
      }
  """
  @spec add_ia3_layers(t(), %{String.t() => module_spec()}) :: t()
  def add_ia3_layers(%__MODULE__{} = model, module_specs) when is_map(module_specs) do
    config = model.config

    ia3_layers =
      Enum.reduce(module_specs, model.ia3_layers, fn {module_name, spec}, acc ->
        if should_apply_ia3?(config, module_name) do
          build_layer(acc, module_name, spec, config)
        else
          acc
        end
      end)

    %{model | ia3_layers: ia3_layers}
  end

  defp build_layer(
         acc,
         module_name,
         %{type: :linear, in_features: in_features, out_features: out_features},
         config
       ) do
    is_feedforward = feedforward_modules_match?(config, module_name)

    layer =
      Linear.new(in_features, out_features,
        config: config,
        is_feedforward: is_feedforward
      )

    Map.put(acc, module_name, layer)
  end

  defp build_layer(acc, _module_name, _spec, _config), do: acc

  @doc """
  Sets the active adapter name.
  """
  @spec set_adapter(t(), String.t()) :: t()
  def set_adapter(%__MODULE__{} = model, adapter_name) do
    %{model | adapter_name: adapter_name}
  end

  @doc """
  Enables the adapter.
  """
  @spec enable_adapter(t(), String.t()) :: t()
  def enable_adapter(%__MODULE__{} = model, _adapter_name) do
    %{model | adapters_enabled: true}
  end

  @doc """
  Disables the adapter.
  """
  @spec disable_adapter(t(), String.t()) :: t()
  def disable_adapter(%__MODULE__{} = model, _adapter_name) do
    %{model | adapters_enabled: false}
  end

  @doc """
  Returns the PEFT configuration for this model.
  """
  @spec get_peft_config(t()) :: IA3Config.t()
  def get_peft_config(%__MODULE__{config: config}), do: config

  @doc """
  Merges the IA3 adapter weights into the base model weights.

  After merging, the model can be used for inference without
  computing the IA3 contribution separately.
  """
  @spec merge_adapter(t()) :: t()
  def merge_adapter(%__MODULE__{merged: true} = model), do: model

  def merge_adapter(%__MODULE__{} = model) do
    %{model | merged: true}
  end

  @doc """
  Unmerges the IA3 adapter weights from the base model weights.

  After unmerging, the IA3 contribution will be computed separately
  during the forward pass.
  """
  @spec unmerge_adapter(t()) :: t()
  def unmerge_adapter(%__MODULE__{merged: false} = model), do: model

  def unmerge_adapter(%__MODULE__{} = model) do
    %{model | merged: false}
  end

  @doc """
  Returns a map of trainable parameters.

  For IA3, this includes only the `ia3_l` scaling vectors for each layer.
  """
  @spec get_trainable_params(t()) :: map()
  def get_trainable_params(%__MODULE__{ia3_layers: ia3_layers}) do
    Map.new(ia3_layers, fn {module_name, layer} ->
      {"#{module_name}.ia3_l", layer.ia3_l}
    end)
  end

  @doc """
  Applies IA3 adapters to a set of module outputs.

  The `module_inputs` map should include `:input` and `:base_output` keys.
  """
  @spec apply_adapters(t(), %{String.t() => map()}, keyword()) :: %{String.t() => Nx.Tensor.t()}
  def apply_adapters(%__MODULE__{} = model, module_inputs, _opts \\ [])
      when is_map(module_inputs) do
    if not model.adapters_enabled or model.merged do
      Map.new(module_inputs, fn {module_name, input_spec} ->
        {module_name, Map.fetch!(input_spec, :base_output)}
      end)
    else
      Enum.reduce(module_inputs, %{}, fn {module_name, input_spec}, acc ->
        base_output = Map.fetch!(input_spec, :base_output)
        layer = Map.get(model.ia3_layers, module_name)
        output = apply_layer(layer, input_spec, base_output)
        Map.put(acc, module_name, output)
      end)
    end
  end

  defp apply_layer(nil, _input_spec, base_output), do: base_output

  defp apply_layer(%Linear{} = layer, input_spec, base_output) do
    x = Map.fetch!(input_spec, :input)
    Linear.forward(layer, x, base_output)
  end

  defp apply_layer(_layer, _input_spec, base_output), do: base_output
end
