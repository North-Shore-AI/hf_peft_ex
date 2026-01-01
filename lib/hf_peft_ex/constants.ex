defmodule HfPeftEx.Constants do
  @moduledoc """
  Model-to-target-module mappings and configuration constants.

  This module provides default target module mappings for various model architectures
  and PEFT methods, as well as standard file names and constants used throughout
  the library.

  ## Target Module Mappings

  When applying PEFT methods like LoRA, you need to specify which modules (layers)
  to adapt. This module provides sensible defaults for common model architectures.

  ## Example

      # Get default LoRA target modules for LLaMA
      HfPeftEx.Constants.get_lora_target_modules("llama")
      #=> ["q_proj", "v_proj"]

      # Get IA3 target modules for GPT-2
      HfPeftEx.Constants.get_ia3_target_modules("gpt2")
      #=> ["c_attn", "mlp.c_proj"]

  ## File Constants

      HfPeftEx.Constants.config_name()
      #=> "adapter_config.json"

      HfPeftEx.Constants.safetensors_weights_name()
      #=> "adapter_model.safetensors"

  """

  # =============================================================================
  # File Name Constants
  # =============================================================================

  @config_name "adapter_config.json"
  @weights_name "adapter_model.bin"
  @safetensors_weights_name "adapter_model.safetensors"
  @tokenizer_config_name "tokenizer_config.json"

  # =============================================================================
  # Special Values
  # =============================================================================

  @all_linear_shorthand "all-linear"
  @min_target_modules_for_optimization 20

  # =============================================================================
  # Layer Name Constants
  # =============================================================================

  @embedding_layer_names ["embed_tokens", "lm_head"]
  @seq_cls_head_names ["score", "classifier"]

  # =============================================================================
  # LoRA Target Modules Mapping
  # Based on: peft/src/peft/utils/constants.py
  # =============================================================================

  @lora_target_modules %{
    "t5" => ["q", "v"],
    "mt5" => ["q", "v"],
    "bart" => ["q_proj", "v_proj"],
    "gpt2" => ["c_attn"],
    "bloom" => ["query_key_value"],
    "blip-2" => ["q", "v", "q_proj", "v_proj"],
    "opt" => ["q_proj", "v_proj"],
    "gptj" => ["q_proj", "v_proj"],
    "gpt_neox" => ["query_key_value"],
    "gpt_neo" => ["q_proj", "v_proj"],
    "bert" => ["query", "value"],
    "roberta" => ["query", "value"],
    "xlm-roberta" => ["query", "value"],
    "electra" => ["query", "value"],
    "deberta-v2" => ["query_proj", "value_proj"],
    "deberta" => ["in_proj"],
    "layoutlm" => ["query", "value"],
    "llama" => ["q_proj", "v_proj"],
    "llama4" => ["q_proj", "v_proj"],
    "chatglm" => ["query_key_value"],
    "gpt_bigcode" => ["c_attn"],
    "mpt" => ["Wqkv"],
    "RefinedWebModel" => ["query_key_value"],
    "RefinedWeb" => ["query_key_value"],
    "falcon" => ["query_key_value"],
    "btlm" => ["c_proj", "c_attn"],
    "codegen" => ["qkv_proj"],
    "mistral" => ["q_proj", "v_proj"],
    "mixtral" => ["q_proj", "v_proj"],
    "stablelm" => ["q_proj", "v_proj"],
    "phi" => ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma" => ["q_proj", "v_proj"],
    "gemma2" => ["q_proj", "v_proj"],
    "gemma3_text" => ["q_proj", "v_proj"],
    "qwen2" => ["q_proj", "v_proj"],
    "qwen3" => ["q_proj", "v_proj"],
    "rwkv" => ["key", "value", "receptance", "output"],
    "rwkv7" => ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]
  }

  # =============================================================================
  # IA3 Target Modules Mapping
  # =============================================================================

  @ia3_target_modules %{
    "t5" => ["k", "v", "wo"],
    "mt5" => ["k", "v", "wi_1"],
    "gpt2" => ["c_attn", "mlp.c_proj"],
    "bloom" => ["query_key_value", "mlp.dense_4h_to_h"],
    "roberta" => ["key", "value", "output.dense"],
    "opt" => ["q_proj", "k_proj", "fc2"],
    "gptj" => ["q_proj", "v_proj", "fc_out"],
    "gpt_neox" => ["query_key_value", "dense_4h_to_h"],
    "gpt_neo" => ["q_proj", "v_proj", "c_proj"],
    "bart" => ["q_proj", "v_proj", "fc2"],
    "gpt_bigcode" => ["c_attn", "mlp.c_proj"],
    "llama" => ["k_proj", "v_proj", "down_proj"],
    "llama4" => ["q_proj", "v_proj", "down_proj"],
    "mistral" => ["k_proj", "v_proj", "down_proj"],
    "mixtral" => ["k_proj", "v_proj", "w2"],
    "bert" => ["key", "value", "output.dense"],
    "deberta-v2" => ["key_proj", "value_proj", "output.dense"],
    "deberta" => ["in_proj", "output.dense"],
    "RefinedWebModel" => ["query_key_value", "dense_4h_to_h"],
    "RefinedWeb" => ["query_key_value", "dense_4h_to_h"],
    "falcon" => ["query_key_value", "dense_4h_to_h"],
    "phi" => ["q_proj", "v_proj", "fc2"],
    "gemma" => ["q_proj", "v_proj", "down_proj"],
    "gemma2" => ["q_proj", "v_proj", "down_proj"],
    "gemma3_text" => ["q_proj", "v_proj", "down_proj"],
    "qwen2" => ["q_proj", "v_proj", "down_proj"],
    "qwen3" => ["q_proj", "v_proj", "down_proj"]
  }

  # =============================================================================
  # IA3 Feedforward Modules Mapping
  # =============================================================================

  @ia3_feedforward_modules %{
    "t5" => ["wo"],
    "mt5" => [],
    "gpt2" => ["mlp.c_proj"],
    "bloom" => ["mlp.dense_4h_to_h"],
    "roberta" => ["output.dense"],
    "opt" => ["fc2"],
    "gptj" => ["fc_out"],
    "gpt_neox" => ["dense_4h_to_h"],
    "gpt_neo" => ["c_proj"],
    "bart" => ["fc2"],
    "gpt_bigcode" => ["mlp.c_proj"],
    "llama" => ["down_proj"],
    "llama4" => ["down_proj"],
    "mistral" => ["down_proj"],
    "mixtral" => ["w2"],
    "bert" => ["output.dense"],
    "deberta-v2" => ["output.dense"],
    "deberta" => ["output.dense"],
    "RefinedWeb" => ["dense_4h_to_h"],
    "RefinedWebModel" => ["dense_4h_to_h"],
    "falcon" => ["dense_4h_to_h"],
    "phi" => ["fc2"],
    "gemma" => ["down_proj"],
    "gemma2" => ["down_proj"],
    "gemma3_text" => ["down_proj"],
    "qwen2" => ["down_proj"],
    "qwen3" => ["down_proj"]
  }

  # =============================================================================
  # AdaLoRA Target Modules Mapping
  # =============================================================================

  @adalora_target_modules %{
    "t5" => ["q", "k", "v", "o", "wi", "wo"],
    "mt5" => ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart" => ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gpt2" => ["c_attn"],
    "bloom" => ["query_key_value"],
    "opt" => ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gptj" => ["q_proj", "v_proj"],
    "gpt_neox" => ["query_key_value"],
    "gpt_neo" => ["q_proj", "v_proj"],
    "llama" => ["q_proj", "v_proj"],
    "llama4" => ["q_proj", "v_proj"],
    "bert" => ["query", "value"],
    "roberta" => ["query", "key", "value", "dense"],
    "deberta-v2" => ["query_proj", "key_proj", "value_proj", "dense"],
    "gpt_bigcode" => ["c_attn"],
    "deberta" => ["in_proj"],
    "gemma" => ["q_proj", "v_proj"],
    "gemma2" => ["q_proj", "v_proj"],
    "gemma3_text" => ["q_proj", "v_proj"],
    "qwen2" => ["q_proj", "v_proj"],
    "qwen3" => ["q_proj", "v_proj"]
  }

  # =============================================================================
  # Public API - File Names
  # =============================================================================

  @doc """
  Returns the standard adapter configuration file name.

  ## Example

      iex> HfPeftEx.Constants.config_name()
      "adapter_config.json"

  """
  @spec config_name() :: String.t()
  def config_name, do: @config_name

  @doc """
  Returns the standard adapter weights file name (PyTorch format).

  ## Example

      iex> HfPeftEx.Constants.weights_name()
      "adapter_model.bin"

  """
  @spec weights_name() :: String.t()
  def weights_name, do: @weights_name

  @doc """
  Returns the standard adapter weights file name (safetensors format).

  Safetensors is the preferred format for cross-platform compatibility.

  ## Example

      iex> HfPeftEx.Constants.safetensors_weights_name()
      "adapter_model.safetensors"

  """
  @spec safetensors_weights_name() :: String.t()
  def safetensors_weights_name, do: @safetensors_weights_name

  @doc """
  Returns the standard tokenizer configuration file name.

  ## Example

      iex> HfPeftEx.Constants.tokenizer_config_name()
      "tokenizer_config.json"

  """
  @spec tokenizer_config_name() :: String.t()
  def tokenizer_config_name, do: @tokenizer_config_name

  # =============================================================================
  # Public API - Special Values
  # =============================================================================

  @doc """
  Returns the shorthand string for targeting all linear layers.

  When passed as `target_modules`, all linear layers in the model will be
  adapted with the PEFT method.

  ## Example

      iex> HfPeftEx.Constants.all_linear_shorthand()
      "all-linear"

  """
  @spec all_linear_shorthand() :: String.t()
  def all_linear_shorthand, do: @all_linear_shorthand

  @doc """
  Returns the threshold for target modules optimization.

  When users specify more than this number of target modules, an optimization
  is applied to reduce the target modules to a minimal set of suffixes.

  ## Example

      iex> HfPeftEx.Constants.min_target_modules_for_optimization()
      20

  """
  @spec min_target_modules_for_optimization() :: pos_integer()
  def min_target_modules_for_optimization, do: @min_target_modules_for_optimization

  # =============================================================================
  # Public API - Layer Names
  # =============================================================================

  @doc """
  Returns the list of common embedding layer names.

  These are typically the input and output embedding layers that may need
  special handling during PEFT.

  ## Example

      iex> HfPeftEx.Constants.embedding_layer_names()
      ["embed_tokens", "lm_head"]

  """
  @spec embedding_layer_names() :: [String.t()]
  def embedding_layer_names, do: @embedding_layer_names

  @doc """
  Returns the list of common sequence classification head names.

  ## Example

      iex> HfPeftEx.Constants.seq_cls_head_names()
      ["score", "classifier"]

  """
  @spec seq_cls_head_names() :: [String.t()]
  def seq_cls_head_names, do: @seq_cls_head_names

  # =============================================================================
  # Public API - LoRA Target Modules
  # =============================================================================

  @doc """
  Returns the default LoRA target modules for a given model type.

  These are the recommended modules to apply LoRA to for each model architecture.
  Returns `nil` if the model type is not recognized.

  ## Examples

      iex> HfPeftEx.Constants.get_lora_target_modules("llama")
      ["q_proj", "v_proj"]

      iex> HfPeftEx.Constants.get_lora_target_modules("bert")
      ["query", "value"]

      iex> HfPeftEx.Constants.get_lora_target_modules("unknown")
      nil

  """
  @spec get_lora_target_modules(String.t()) :: [String.t()] | nil
  def get_lora_target_modules(model_type) when is_binary(model_type) do
    Map.get(@lora_target_modules, model_type)
  end

  # =============================================================================
  # Public API - IA3 Target Modules
  # =============================================================================

  @doc """
  Returns the default IA3 target modules for a given model type.

  IA3 typically targets key, value, and feedforward layers.
  Returns `nil` if the model type is not recognized.

  ## Examples

      iex> HfPeftEx.Constants.get_ia3_target_modules("llama")
      ["k_proj", "v_proj", "down_proj"]

      iex> HfPeftEx.Constants.get_ia3_target_modules("gpt2")
      ["c_attn", "mlp.c_proj"]

  """
  @spec get_ia3_target_modules(String.t()) :: [String.t()] | nil
  def get_ia3_target_modules(model_type) when is_binary(model_type) do
    Map.get(@ia3_target_modules, model_type)
  end

  @doc """
  Returns the default IA3 feedforward modules for a given model type.

  These are the feedforward/MLP modules within the IA3 target modules.
  Returns `nil` if the model type is not recognized.

  ## Examples

      iex> HfPeftEx.Constants.get_ia3_feedforward_modules("llama")
      ["down_proj"]

      iex> HfPeftEx.Constants.get_ia3_feedforward_modules("gpt2")
      ["mlp.c_proj"]

  """
  @spec get_ia3_feedforward_modules(String.t()) :: [String.t()] | nil
  def get_ia3_feedforward_modules(model_type) when is_binary(model_type) do
    Map.get(@ia3_feedforward_modules, model_type)
  end

  # =============================================================================
  # Public API - AdaLoRA Target Modules
  # =============================================================================

  @doc """
  Returns the default AdaLoRA target modules for a given model type.

  AdaLoRA typically targets more modules than standard LoRA for adaptive
  rank allocation.
  Returns `nil` if the model type is not recognized.

  ## Examples

      iex> HfPeftEx.Constants.get_adalora_target_modules("llama")
      ["q_proj", "v_proj"]

      iex> HfPeftEx.Constants.get_adalora_target_modules("t5")
      ["q", "k", "v", "o", "wi", "wo"]

  """
  @spec get_adalora_target_modules(String.t()) :: [String.t()] | nil
  def get_adalora_target_modules(model_type) when is_binary(model_type) do
    Map.get(@adalora_target_modules, model_type)
  end

  # =============================================================================
  # Public API - Generic Target Modules
  # =============================================================================

  @doc """
  Returns the default target modules for a PEFT type and model type.

  This is a convenience function that dispatches to the appropriate
  method-specific lookup function.

  ## Examples

      iex> HfPeftEx.Constants.get_target_modules(:lora, "llama")
      ["q_proj", "v_proj"]

      iex> HfPeftEx.Constants.get_target_modules(:ia3, "gpt2")
      ["c_attn", "mlp.c_proj"]

      iex> HfPeftEx.Constants.get_target_modules(:unknown, "llama")
      nil

  """
  @spec get_target_modules(atom(), String.t()) :: [String.t()] | nil
  def get_target_modules(peft_type, model_type)
      when is_atom(peft_type) and is_binary(model_type) do
    case peft_type do
      :lora ->
        get_lora_target_modules(model_type)

      :adalora ->
        get_adalora_target_modules(model_type)

      :ia3 ->
        get_ia3_target_modules(model_type)

      # Many PEFT methods share the same target modules as LoRA
      type when type in [:loha, :lokr, :oft, :boft, :poly, :bone, :randlora, :road, :delora, :hra] ->
        get_lora_target_modules(model_type)

      _ ->
        nil
    end
  end

  # =============================================================================
  # Public API - Supported Model Types
  # =============================================================================

  @doc """
  Returns the list of supported model types for a given PEFT method.

  ## Examples

      iex> types = HfPeftEx.Constants.supported_model_types(:lora)
      iex> "llama" in types
      true

      iex> HfPeftEx.Constants.supported_model_types(:unsupported)
      []

  """
  @spec supported_model_types(atom()) :: [String.t()]
  def supported_model_types(peft_type) when is_atom(peft_type) do
    case peft_type do
      :lora ->
        Map.keys(@lora_target_modules)

      :adalora ->
        Map.keys(@adalora_target_modules)

      :ia3 ->
        Map.keys(@ia3_target_modules)

      type
      when type in [:loha, :lokr, :oft, :boft, :poly, :bone, :randlora, :road, :delora, :hra] ->
        Map.keys(@lora_target_modules)

      _ ->
        []
    end
  end
end
