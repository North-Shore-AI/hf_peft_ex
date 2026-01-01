defmodule HfPeftEx.Mapping do
  @moduledoc """
  Registry pattern for PEFT method type-to-implementation mappings.

  This module provides the mapping between PEFT types (like `:lora`, `:ia3`) and
  their corresponding configuration classes and tuner (model) classes.

  ## Key Functions

  - `get_config_class/1` - Get the config module for a PEFT type
  - `get_tuner_class/1` - Get the tuner/model module for a PEFT type
  - `get_prefix/1` - Get the weight prefix (e.g., "lora_") for a PEFT type
  - `get_peft_config/1` - Factory function to create config from a dict
  - `supported_peft_types/0` - List all supported PEFT types

  ## Example

      iex> HfPeftEx.Mapping.get_config_class(:lora)
      HfPeftEx.Tuners.Lora.Config

      iex> HfPeftEx.Mapping.get_prefix(:lora)
      "lora_"

      iex> HfPeftEx.Mapping.get_peft_config(%{"peft_type" => "lora", "r" => 8})
      {:ok, %HfPeftEx.Tuners.Lora.Config{r: 8, ...}}

  """

  alias HfPeftEx.PeftType

  # Type to config class mapping
  # These are the implemented PEFT methods with their config modules
  @peft_type_to_config %{
    lora: HfPeftEx.Tuners.Lora.Config,
    # Future implementations - use base config as placeholder
    adalora: HfPeftEx.Tuners.Lora.Config,
    ia3: HfPeftEx.Config,
    prefix_tuning: HfPeftEx.Config,
    prompt_tuning: HfPeftEx.Config,
    p_tuning: HfPeftEx.Config,
    multitask_prompt_tuning: HfPeftEx.Config,
    loha: HfPeftEx.Config,
    lokr: HfPeftEx.Config,
    oft: HfPeftEx.Config,
    boft: HfPeftEx.Config,
    poly: HfPeftEx.Config,
    ln_tuning: HfPeftEx.Config,
    vera: HfPeftEx.Config,
    fourierft: HfPeftEx.Config,
    xlora: HfPeftEx.Config,
    hra: HfPeftEx.Config,
    vblora: HfPeftEx.Config,
    cpt: HfPeftEx.Config,
    bone: HfPeftEx.Config,
    miss: HfPeftEx.Config,
    randlora: HfPeftEx.Config,
    road: HfPeftEx.Config,
    trainable_tokens: HfPeftEx.Config,
    shira: HfPeftEx.Config,
    c3a: HfPeftEx.Config,
    waveft: HfPeftEx.Config,
    osf: HfPeftEx.Config,
    delora: HfPeftEx.Config,
    gralora: HfPeftEx.Config,
    adaption_prompt: HfPeftEx.Config
  }

  # Type to tuner/model class mapping
  @peft_type_to_tuner %{
    lora: HfPeftEx.Tuners.Lora.Model,
    # Placeholder for future implementations
    adalora: nil,
    ia3: nil,
    prefix_tuning: nil,
    prompt_tuning: nil,
    p_tuning: nil,
    multitask_prompt_tuning: nil,
    loha: nil,
    lokr: nil,
    oft: nil,
    boft: nil,
    poly: nil,
    ln_tuning: nil,
    vera: nil,
    fourierft: nil,
    xlora: nil,
    hra: nil,
    vblora: nil,
    cpt: nil,
    bone: nil,
    miss: nil,
    randlora: nil,
    road: nil,
    trainable_tokens: nil,
    shira: nil,
    c3a: nil,
    waveft: nil,
    osf: nil,
    delora: nil,
    gralora: nil,
    adaption_prompt: nil
  }

  # Type to weight prefix mapping
  @peft_type_to_prefix %{
    lora: "lora_",
    adalora: "lora_",
    ia3: "ia3_",
    prefix_tuning: "prefix_",
    prompt_tuning: "prompt_",
    p_tuning: "p_tuning_",
    multitask_prompt_tuning: "multitask_prompt_",
    loha: "hada_",
    lokr: "lokr_",
    oft: "oft_",
    boft: "boft_",
    poly: "poly_",
    ln_tuning: "ln_tuning_",
    vera: "vera_",
    fourierft: "fourierft_",
    xlora: "xlora_",
    hra: "hra_",
    vblora: "vblora_",
    cpt: "cpt_",
    bone: "bone_",
    miss: "miss_",
    randlora: "randlora_",
    road: "road_",
    trainable_tokens: "trainable_tokens_",
    shira: "shira_",
    c3a: "c3a_",
    waveft: "waveft_",
    osf: "osf_",
    delora: "delora_",
    gralora: "gralora_",
    adaption_prompt: "adaption_"
  }

  @type peft_type :: PeftType.t()
  @type config_class :: module()
  @type tuner_class :: module()

  @doc """
  Returns the configuration class for a given PEFT type.

  Returns `nil` if the PEFT type is unknown.

  ## Examples

      iex> HfPeftEx.Mapping.get_config_class(:lora)
      HfPeftEx.Tuners.Lora.Config

      iex> HfPeftEx.Mapping.get_config_class(:unknown)
      nil

  """
  @spec get_config_class(peft_type()) :: config_class() | nil
  def get_config_class(peft_type) when is_atom(peft_type) do
    Map.get(@peft_type_to_config, peft_type)
  end

  @doc """
  Returns the tuner/model class for a given PEFT type.

  Returns `nil` if the PEFT type is unknown or not yet implemented.

  ## Examples

      iex> HfPeftEx.Mapping.get_tuner_class(:lora)
      HfPeftEx.Tuners.Lora.Model

      iex> HfPeftEx.Mapping.get_tuner_class(:unknown)
      nil

  """
  @spec get_tuner_class(peft_type()) :: tuner_class() | nil
  def get_tuner_class(peft_type) when is_atom(peft_type) do
    Map.get(@peft_type_to_tuner, peft_type)
  end

  @doc """
  Returns the weight prefix for a given PEFT type.

  The prefix is used to identify adapter weights in state dictionaries.
  For example, LoRA weights use the prefix "lora_".

  ## Examples

      iex> HfPeftEx.Mapping.get_prefix(:lora)
      "lora_"

      iex> HfPeftEx.Mapping.get_prefix(:ia3)
      "ia3_"

      iex> HfPeftEx.Mapping.get_prefix(:unknown)
      nil

  """
  @spec get_prefix(peft_type()) :: String.t() | nil
  def get_prefix(peft_type) when is_atom(peft_type) do
    Map.get(@peft_type_to_prefix, peft_type)
  end

  @doc """
  Creates a PEFT configuration from a dictionary.

  This factory function determines the appropriate config class based on
  the `peft_type` key in the dictionary and instantiates it.

  ## Parameters

  - `config_dict` - A map containing configuration values. Must include
    `peft_type` (either as string or atom key, with string or atom value).

  ## Examples

      iex> HfPeftEx.Mapping.get_peft_config(%{"peft_type" => "lora", "r" => 8})
      {:ok, %HfPeftEx.Tuners.Lora.Config{r: 8, ...}}

      iex> HfPeftEx.Mapping.get_peft_config(%{peft_type: :lora, r: 16})
      {:ok, %HfPeftEx.Tuners.Lora.Config{r: 16, ...}}

      iex> HfPeftEx.Mapping.get_peft_config(%{"r" => 8})
      {:error, "Missing peft_type in config dict"}

  """
  @spec get_peft_config(map()) :: {:ok, struct()} | {:error, String.t()}
  def get_peft_config(config_dict) when is_map(config_dict) do
    with {:ok, peft_type} <- extract_peft_type(config_dict),
         {:ok, config_class} <- get_config_class_or_error(peft_type) do
      opts = convert_dict_to_opts(config_dict, peft_type)
      {:ok, config_class.new(opts)}
    end
  end

  @doc """
  Returns a list of all supported PEFT types.

  These are the PEFT types that have configuration mappings registered.

  ## Examples

      iex> types = HfPeftEx.Mapping.supported_peft_types()
      iex> :lora in types
      true

  """
  @spec supported_peft_types() :: [peft_type()]
  def supported_peft_types do
    Map.keys(@peft_type_to_config)
  end

  @doc """
  Injects PEFT adapter layers into a model in-place.

  This function creates PEFT layers and injects them into the model.
  Unlike wrapping with PeftModel, this mutates the model directly.

  Note: Currently only supports LoRA-based methods. Prompt learning
  methods are not supported.

  ## Parameters

  - `peft_config` - The PEFT configuration struct
  - `model` - The model to inject adapters into
  - `adapter_name` - Name for the adapter (default: "default")

  ## Examples

      config = HfPeftEx.Tuners.Lora.Config.new(r: 8)
      {:ok, model} = HfPeftEx.Mapping.inject_adapter_in_model(config, model, "default")

  """
  @spec inject_adapter_in_model(struct(), map(), String.t()) ::
          {:ok, map()} | {:error, String.t()}
  def inject_adapter_in_model(peft_config, model, adapter_name \\ "default")

  def inject_adapter_in_model(%{peft_type: peft_type} = _config, _model, _adapter_name)
      when peft_type in [
             :prompt_tuning,
             :prefix_tuning,
             :p_tuning,
             :multitask_prompt_tuning,
             :cpt,
             :adaption_prompt
           ] do
    {:error, "inject_adapter_in_model does not support prompt learning methods"}
  end

  def inject_adapter_in_model(%{peft_type: peft_type} = _config, _model, _adapter_name) do
    tuner_class = get_tuner_class(peft_type)

    if is_nil(tuner_class) do
      {:error, "Unknown or unsupported PEFT type: #{inspect(peft_type)}"}
    else
      # Injection requires Axon integration - see PEFT_GAP_ANALYSIS.md for roadmap
      {:error, "inject_adapter_in_model not yet implemented for #{inspect(peft_type)}"}
    end
  end

  def inject_adapter_in_model(_config, _model, _adapter_name) do
    {:error, "Invalid config: missing peft_type"}
  end

  # Private helpers

  defp extract_peft_type(config_dict) do
    peft_type =
      Map.get(config_dict, "peft_type") ||
        Map.get(config_dict, :peft_type)

    case peft_type do
      nil ->
        {:error, "Missing peft_type in config dict"}

      type when is_binary(type) ->
        type
        |> String.downcase()
        |> String.to_atom()
        |> validate_peft_type()

      type when is_atom(type) ->
        validate_peft_type(type)
    end
  end

  defp validate_peft_type(peft_type) do
    if PeftType.valid?(peft_type) do
      {:ok, peft_type}
    else
      {:error, "Unknown PEFT type: #{inspect(peft_type)}"}
    end
  end

  defp get_config_class_or_error(peft_type) do
    case get_config_class(peft_type) do
      nil -> {:error, "No config class registered for PEFT type: #{inspect(peft_type)}"}
      class -> {:ok, class}
    end
  end

  defp convert_dict_to_opts(config_dict, _peft_type) do
    config_dict
    |> Enum.map(fn
      {"peft_type", _} -> nil
      {:peft_type, _} -> nil
      {key, value} when is_binary(key) -> {safe_to_atom(key), value}
      {key, value} when is_atom(key) -> {key, value}
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp safe_to_atom(string) do
    String.to_existing_atom(string)
  rescue
    ArgumentError -> String.to_atom(string)
  end
end
