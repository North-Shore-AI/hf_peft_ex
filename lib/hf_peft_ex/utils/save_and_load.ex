defmodule HfPeftEx.Utils.SaveAndLoad do
  @moduledoc """
  Utilities for saving and loading PEFT adapter state dictionaries.

  This module provides functions for extracting adapter weights from PEFT models,
  saving them to files, and loading them back. It handles adapter name prefixing,
  bias modes, and various file formats.

  ## Key Functions

  - `get_peft_model_state_dict/3` - Extract adapter weights from a model
  - `set_peft_model_state_dict/4` - Load weights into a model
  - `save_peft_weights/3` - Save weights to a file
  - `load_peft_weights/2` - Load weights from a file

  ## Example

      # Extract and save weights
      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)
      :ok = SaveAndLoad.save_peft_weights(state_dict, "adapter_model.nx")

      # Load weights back
      {:ok, weights} = SaveAndLoad.load_peft_weights("adapter_model.nx")
      {:ok, model} = SaveAndLoad.set_peft_model_state_dict(model, weights)

  """

  alias HfPeftEx.Mapping

  @type state_dict :: %{String.t() => Nx.Tensor.t()}
  @type adapter_name :: String.t()
  @type peft_type :: atom()

  # =============================================================================
  # Get State Dict
  # =============================================================================

  @doc """
  Extracts PEFT adapter weights from a model.

  This only includes the adapter parameters, not the base model weights.
  The returned state dict uses portable keys without the adapter name,
  suitable for saving and later loading with a different adapter name.

  ## Parameters

  - `model` - The PEFT model containing lora_layers and config
  - `adapter_name` - The adapter to extract (default: "default")
  - `opts` - Additional options (reserved for future use)

  ## Options

  - `:save_embedding_layers` - Whether to include embedding layers (default: :auto)

  ## Examples

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)
      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model, "custom_adapter")

  """
  @spec get_peft_model_state_dict(map(), adapter_name(), keyword()) ::
          {:ok, state_dict()} | {:error, String.t()}
  def get_peft_model_state_dict(model, adapter_name \\ "default", _opts \\ []) do
    config = get_adapter_config(model, adapter_name)

    case config do
      nil ->
        {:error, "Adapter '#{adapter_name}' not found in model"}

      config ->
        state_dict = extract_adapter_state_dict(model, adapter_name, config)
        # Remove adapter name from keys for portable format
        portable_dict = remove_adapter_name_from_keys(state_dict, adapter_name)
        {:ok, portable_dict}
    end
  end

  # =============================================================================
  # Set State Dict
  # =============================================================================

  @doc """
  Loads PEFT weights into a model.

  Takes a portable state dict (without adapter names in keys) and inserts
  the weights into the model under the specified adapter name.

  ## Parameters

  - `model` - The PEFT model to load weights into
  - `state_dict` - The state dict containing adapter weights
  - `adapter_name` - The adapter name to use (default: "default")
  - `opts` - Additional options

  ## Options

  - `:ignore_mismatched_sizes` - Skip tensors with shape mismatches (default: false)

  ## Examples

      {:ok, model} = SaveAndLoad.set_peft_model_state_dict(model, state_dict)
      {:ok, model} = SaveAndLoad.set_peft_model_state_dict(model, state_dict, "custom")

  """
  @spec set_peft_model_state_dict(map(), state_dict(), adapter_name(), keyword()) ::
          {:ok, map()} | {:error, String.t()}
  def set_peft_model_state_dict(model, state_dict, adapter_name \\ "default", opts \\ []) do
    config = get_adapter_config(model, adapter_name)
    ignore_mismatched = Keyword.get(opts, :ignore_mismatched_sizes, false)

    peft_type = if config, do: config.peft_type, else: :lora
    prefix = Mapping.get_prefix(peft_type) || "lora_"

    # Add adapter name to keys
    prefixed_dict = add_adapter_name_to_keys(state_dict, adapter_name, prefix)

    # Load weights into model
    updated_model = load_weights_into_model(model, prefixed_dict, adapter_name, ignore_mismatched)

    {:ok, updated_model}
  end

  # =============================================================================
  # Save Weights
  # =============================================================================

  @doc """
  Saves adapter weights to a file.

  Currently supports Nx serialization format. Safetensors support is planned
  for future releases for cross-platform compatibility with Python.

  ## Parameters

  - `state_dict` - Map of weight names to Nx tensors
  - `path` - File path to save to
  - `opts` - Additional options

  ## Options

  - `:format` - File format, one of :nx, :safetensors (default: :nx)

  ## Examples

      :ok = SaveAndLoad.save_peft_weights(state_dict, "adapter.nx")
      :ok = SaveAndLoad.save_peft_weights(state_dict, "adapter.nx", format: :nx)

  """
  @spec save_peft_weights(state_dict(), String.t(), keyword()) :: :ok | {:error, term()}
  def save_peft_weights(state_dict, path, opts \\ []) do
    format = Keyword.get(opts, :format, :nx)

    case format do
      :nx ->
        save_nx_format(state_dict, path)

      :safetensors ->
        # Safetensors support planned for cross-platform Python compatibility
        {:error, "Safetensors format not yet implemented"}

      other ->
        {:error, "Unknown format: #{inspect(other)}"}
    end
  end

  # =============================================================================
  # Load Weights
  # =============================================================================

  @doc """
  Loads adapter weights from a file.

  Automatically detects the file format based on extension.

  ## Parameters

  - `path` - File path to load from
  - `opts` - Additional options (reserved for future use)

  ## Examples

      {:ok, weights} = SaveAndLoad.load_peft_weights("adapter.nx")

  """
  @spec load_peft_weights(String.t(), keyword()) :: {:ok, state_dict()} | {:error, term()}
  def load_peft_weights(path, _opts \\ []) do
    cond do
      not File.exists?(path) ->
        {:error, "File not found: #{path}"}

      String.ends_with?(path, ".safetensors") ->
        # Safetensors support planned for cross-platform Python compatibility
        {:error, "Safetensors format not yet implemented"}

      true ->
        load_nx_format(path)
    end
  end

  # =============================================================================
  # Filter Keys
  # =============================================================================

  @doc """
  Filters a state dict to only include adapter-related keys.

  This function keeps only keys that:
  1. Contain the PEFT type prefix (e.g., "lora_")
  2. Belong to the specified adapter

  ## Parameters

  - `state_dict` - The full state dict to filter
  - `adapter_name` - The adapter name to filter for
  - `peft_type` - The PEFT type (e.g., :lora, :ia3)

  ## Examples

      filtered = SaveAndLoad.filter_adapter_keys(state_dict, "default", :lora)

  """
  @spec filter_adapter_keys(state_dict(), adapter_name(), peft_type()) :: state_dict()
  def filter_adapter_keys(state_dict, adapter_name, peft_type) do
    prefix = Mapping.get_prefix(peft_type) || "lora_"

    state_dict
    |> Enum.filter(fn {key, _value} ->
      String.contains?(key, prefix) and
        (String.contains?(key, ".#{adapter_name}.") or
           String.ends_with?(key, ".#{adapter_name}"))
    end)
    |> Map.new()
  end

  # =============================================================================
  # Key Manipulation
  # =============================================================================

  @doc """
  Removes the adapter name from state dict keys.

  Converts keys like `layer.lora_A.default.weight` to `layer.lora_A.weight`.
  This creates a portable format that can be loaded with any adapter name.

  ## Parameters

  - `state_dict` - The state dict to transform
  - `adapter_name` - The adapter name to remove

  ## Examples

      portable = SaveAndLoad.remove_adapter_name_from_keys(state_dict, "default")

  """
  @spec remove_adapter_name_from_keys(state_dict(), adapter_name()) :: state_dict()
  def remove_adapter_name_from_keys(state_dict, adapter_name) do
    Map.new(state_dict, fn {key, value} ->
      new_key = remove_adapter_name(key, adapter_name)
      {new_key, value}
    end)
  end

  @doc """
  Adds the adapter name to state dict keys.

  Converts keys like `layer.lora_A.weight` to `layer.lora_A.adapter_name.weight`.
  This is the inverse of `remove_adapter_name_from_keys/2`.

  ## Parameters

  - `state_dict` - The state dict to transform
  - `adapter_name` - The adapter name to add
  - `prefix` - The PEFT prefix to look for (e.g., "lora_")

  ## Examples

      prefixed = SaveAndLoad.add_adapter_name_to_keys(state_dict, "custom", "lora_")

  """
  @spec add_adapter_name_to_keys(state_dict(), adapter_name(), String.t()) :: state_dict()
  def add_adapter_name_to_keys(state_dict, adapter_name, prefix) do
    Map.new(state_dict, fn {key, value} ->
      new_key = add_adapter_name(key, adapter_name, prefix)
      {new_key, value}
    end)
  end

  # =============================================================================
  # Private Helpers
  # =============================================================================

  defp get_adapter_config(model, adapter_name) do
    case Map.get(model, :peft_config) do
      nil -> Map.get(model, :config)
      configs when is_map(configs) -> Map.get(configs, adapter_name)
    end
  end

  defp extract_adapter_state_dict(model, adapter_name, config) do
    lora_layers = Map.get(model, :lora_layers, %{})
    bias_mode = Map.get(config, :bias, :none)

    # Extract weights from each layer
    Enum.reduce(lora_layers, %{}, fn {layer_name, layer}, acc ->
      acc
      |> maybe_add_weight(layer_name, "lora_A", layer, :lora_a, adapter_name)
      |> maybe_add_weight(layer_name, "lora_B", layer, :lora_b, adapter_name)
      |> maybe_add_bias(layer_name, layer, adapter_name, bias_mode)
    end)
  end

  defp maybe_add_weight(acc, layer_name, weight_type, layer, key, adapter_name) do
    case get_in(layer, [key, adapter_name]) do
      nil ->
        acc

      tensor ->
        full_key = "#{layer_name}.#{weight_type}.#{adapter_name}.weight"
        Map.put(acc, full_key, tensor)
    end
  end

  defp maybe_add_bias(acc, layer_name, layer, adapter_name, bias_mode) do
    case bias_mode do
      :none ->
        acc

      mode when mode in [:all, :lora_only] ->
        case get_in(layer, [:lora_bias, adapter_name]) do
          nil -> acc
          tensor -> Map.put(acc, "#{layer_name}.lora_bias.#{adapter_name}", tensor)
        end
    end
  end

  defp remove_adapter_name(key, adapter_name) do
    # Handle pattern: layer.lora_A.adapter_name.weight -> layer.lora_A.weight
    pattern = ".#{adapter_name}."

    if String.contains?(key, pattern) do
      String.replace(key, pattern, ".")
    else
      # Handle pattern: layer.lora_A.adapter_name -> layer.lora_A
      suffix = ".#{adapter_name}"

      if String.ends_with?(key, suffix) do
        String.replace_suffix(key, suffix, "")
      else
        key
      end
    end
  end

  defp add_adapter_name(key, adapter_name, prefix) do
    # Find the prefix (e.g., "lora_A", "lora_B") and insert adapter name after it
    cond do
      String.contains?(key, "#{prefix}A.") or String.contains?(key, "#{prefix}B.") ->
        # Key is like: layer.lora_A.weight -> layer.lora_A.adapter_name.weight
        key
        |> String.replace("#{prefix}A.", "#{prefix}A.#{adapter_name}.")
        |> String.replace("#{prefix}B.", "#{prefix}B.#{adapter_name}.")

      String.contains?(key, "#{prefix}A") or String.contains?(key, "#{prefix}B") ->
        # Key is like: layer.lora_A -> layer.lora_A.adapter_name
        key
        |> maybe_insert_after("#{prefix}A", ".#{adapter_name}")
        |> maybe_insert_after("#{prefix}B", ".#{adapter_name}")

      true ->
        key
    end
  end

  defp maybe_insert_after(string, search, insert) do
    case String.split(string, search, parts: 2) do
      [before, after_part] when after_part != "" ->
        if String.starts_with?(after_part, ".") do
          # Already has something after, insert adapter name
          before <> search <> insert <> after_part
        else
          before <> search <> after_part
        end

      [before, ""] ->
        before <> search <> insert

      _ ->
        string
    end
  end

  defp load_weights_into_model(model, state_dict, adapter_name, ignore_mismatched) do
    lora_layers = Map.get(model, :lora_layers, %{})

    updated_layers =
      Enum.reduce(state_dict, lora_layers, fn {key, tensor}, acc ->
        case parse_weight_key(key) do
          {:ok, layer_name, weight_type, _adapter} ->
            update_layer_weight(
              acc,
              layer_name,
              weight_type,
              adapter_name,
              tensor,
              ignore_mismatched
            )

          :error ->
            acc
        end
      end)

    %{model | lora_layers: updated_layers}
  end

  defp parse_weight_key(key) do
    # Parse keys like: layer1.lora_A.adapter_name.weight
    parts = String.split(key, ".")
    lora_idx = Enum.find_index(parts, &String.starts_with?(&1, "lora_"))

    if lora_idx && lora_idx >= 1 do
      layer_name = parts |> Enum.take(lora_idx) |> Enum.join(".")
      weight_type = Enum.at(parts, lora_idx)
      adapter = Enum.at(parts, lora_idx + 1)
      {:ok, layer_name, weight_type, adapter}
    else
      :error
    end
  end

  defp update_layer_weight(layers, layer_name, weight_type, adapter_name, tensor, opts) do
    key = weight_type_to_key(weight_type)

    case key do
      nil -> layers
      key -> do_update_layer_weight(layers, layer_name, key, adapter_name, tensor, opts)
    end
  end

  defp weight_type_to_key("lora_A"), do: :lora_a
  defp weight_type_to_key("lora_B"), do: :lora_b
  defp weight_type_to_key("lora_bias"), do: :lora_bias
  defp weight_type_to_key(_), do: nil

  defp do_update_layer_weight(layers, layer_name, key, adapter_name, tensor, ignore_mismatched) do
    layer = Map.get(layers, layer_name, %{lora_a: %{}, lora_b: %{}, lora_bias: nil})
    current = Map.get(layer, key) || %{}

    if should_update_weight?(current, adapter_name, tensor, ignore_mismatched) do
      updated_weights = Map.put(current, adapter_name, tensor)
      updated_layer = Map.put(layer, key, updated_weights)
      Map.put(layers, layer_name, updated_layer)
    else
      layers
    end
  end

  defp should_update_weight?(current, adapter_name, tensor, ignore_mismatched) do
    case Map.get(current, adapter_name) do
      nil -> true
      existing -> Nx.shape(existing) == Nx.shape(tensor) or ignore_mismatched
    end
  end

  defp save_nx_format(state_dict, path) do
    # Convert to serializable format
    serializable =
      Map.new(state_dict, fn {key, tensor} ->
        {key, Nx.to_binary(tensor)}
      end)

    # Include shape info for deserialization
    metadata =
      Map.new(state_dict, fn {key, tensor} ->
        {key, %{shape: Nx.shape(tensor), type: Nx.type(tensor)}}
      end)

    data = %{weights: serializable, metadata: metadata}
    binary = :erlang.term_to_binary(data)

    File.write(path, binary)
  end

  defp load_nx_format(path) do
    case File.read(path) do
      {:ok, binary} ->
        try do
          %{weights: weights, metadata: metadata} = :erlang.binary_to_term(binary)

          tensors =
            Map.new(weights, fn {key, binary_data} ->
              %{shape: shape, type: type} = Map.fetch!(metadata, key)
              tensor = Nx.from_binary(binary_data, type) |> Nx.reshape(shape)
              {key, tensor}
            end)

          {:ok, tensors}
        rescue
          e -> {:error, Exception.message(e)}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end
end
