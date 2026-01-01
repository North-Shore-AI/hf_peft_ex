# Constants Module

## Overview

Model-to-target-module mappings for automatic adapter injection. Essential for supporting multiple model architectures without manual configuration.

## Python Reference

**File:** `peft/src/peft/utils/constants.py` (362 lines)

### Key Constants

#### LoRA Target Modules
```python
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": ["q_proj", "v_proj"],
    "gemma2": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
    "qwen2_moe": ["q_proj", "v_proj"],
    "qwen3": ["q_proj", "v_proj"],
    "qwen3_moe": ["q_proj", "v_proj"],
    # ... many more
}
```

#### IA3 Target Modules
```python
TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "t5": ["k", "v", "wo"],
    "mt5": ["k", "v", "wo"],
    "gpt2": ["c_attn", "mlp.c_proj"],
    "bloom": ["query_key_value", "mlp.dense_4h_to_h"],
    "llama": ["k_proj", "v_proj", "down_proj"],
    # ...
}

TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {
    "t5": ["wo"],
    "mt5": ["wo"],
    "gpt2": ["mlp.c_proj"],
    "bloom": ["mlp.dense_4h_to_h"],
    "llama": ["down_proj"],
    # ...
}
```

#### Other Method Mappings
```python
TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_BOFT_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_LOHA_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_LOKR_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_POLY_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_BONE_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_ROAD_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING = {...}
TRANSFORMERS_MODELS_TO_DELORA_TARGET_MODULES_MAPPING = {...}
```

#### Special Constants
```python
EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]
SEQ_CLS_HEAD_NAMES = ["classifier", "score"]
DUMMY_MODEL_CONFIG = PretrainedConfig(...)
DUMMY_TARGET_MODULES = "all-linear"
MIN_TARGET_MODULES_FOR_OPTIMIZATION = 16
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"
```

## Elixir Implementation Design

### Module: `HfPeftEx.Constants`

```elixir
defmodule HfPeftEx.Constants do
  @moduledoc """
  Model-to-target-module mappings and configuration constants.
  """

  # File names
  @config_name "adapter_config.json"
  @weights_name "adapter_model.bin"
  @safetensors_weights_name "adapter_model.safetensors"

  # Special values
  @all_linear_shorthand "all-linear"
  @min_target_modules_for_optimization 16

  # Embedding layer names
  @embedding_layer_names ["embed_tokens", "lm_head"]

  # Classification head names
  @seq_cls_head_names ["classifier", "score"]

  @lora_target_modules %{
    "t5" => ["q", "v"],
    "mt5" => ["q", "v"],
    "bart" => ["q_proj", "v_proj"],
    "gpt2" => ["c_attn"],
    "bloom" => ["query_key_value"],
    "opt" => ["q_proj", "v_proj"],
    "gptj" => ["q_proj", "v_proj"],
    "gpt_neox" => ["query_key_value"],
    "bert" => ["query", "value"],
    "roberta" => ["query", "value"],
    "llama" => ["q_proj", "v_proj"],
    "mistral" => ["q_proj", "v_proj"],
    "mixtral" => ["q_proj", "v_proj"],
    "phi" => ["q_proj", "v_proj", "fc1", "fc2"],
    "phi3" => ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma" => ["q_proj", "v_proj"],
    "gemma2" => ["q_proj", "v_proj"],
    "qwen2" => ["q_proj", "v_proj"],
    "qwen3" => ["q_proj", "v_proj"],
    "falcon" => ["query_key_value"],
    "mpt" => ["Wqkv"],
    "stablelm" => ["q_proj", "v_proj"]
  }

  @ia3_target_modules %{
    "t5" => ["k", "v", "wo"],
    "gpt2" => ["c_attn", "mlp.c_proj"],
    "bloom" => ["query_key_value", "mlp.dense_4h_to_h"],
    "llama" => ["k_proj", "v_proj", "down_proj"],
    "opt" => ["k_proj", "v_proj", "fc2"]
  }

  @ia3_feedforward_modules %{
    "t5" => ["wo"],
    "gpt2" => ["mlp.c_proj"],
    "bloom" => ["mlp.dense_4h_to_h"],
    "llama" => ["down_proj"],
    "opt" => ["fc2"]
  }

  # Public API
  @spec config_name() :: String.t()
  def config_name, do: @config_name

  @spec weights_name() :: String.t()
  def weights_name, do: @weights_name

  @spec safetensors_weights_name() :: String.t()
  def safetensors_weights_name, do: @safetensors_weights_name

  @spec all_linear_shorthand() :: String.t()
  def all_linear_shorthand, do: @all_linear_shorthand

  @spec embedding_layer_names() :: [String.t()]
  def embedding_layer_names, do: @embedding_layer_names

  @spec get_lora_target_modules(String.t()) :: [String.t()] | nil
  def get_lora_target_modules(model_type) do
    Map.get(@lora_target_modules, model_type)
  end

  @spec get_ia3_target_modules(String.t()) :: [String.t()] | nil
  def get_ia3_target_modules(model_type) do
    Map.get(@ia3_target_modules, model_type)
  end

  @spec get_ia3_feedforward_modules(String.t()) :: [String.t()] | nil
  def get_ia3_feedforward_modules(model_type) do
    Map.get(@ia3_feedforward_modules, model_type)
  end

  @spec get_target_modules(atom(), String.t()) :: [String.t()] | nil
  def get_target_modules(peft_type, model_type) do
    case peft_type do
      :lora -> get_lora_target_modules(model_type)
      :adalora -> get_lora_target_modules(model_type)  # Same as LoRA
      :ia3 -> get_ia3_target_modules(model_type)
      # Add more as implemented
      _ -> nil
    end
  end

  @spec supported_model_types(atom()) :: [String.t()]
  def supported_model_types(:lora), do: Map.keys(@lora_target_modules)
  def supported_model_types(:ia3), do: Map.keys(@ia3_target_modules)
  def supported_model_types(_), do: []
end
```

## Files to Read

- `peft/src/peft/utils/constants.py` (full file)
- `lib/hf_peft_ex/tuners/lora/config.ex` (target_modules usage)

## Tests Required

1. Lookup LoRA target modules for all supported models
2. Lookup IA3 target modules
3. Lookup feedforward modules
4. Handle unknown model types (returns nil)
5. Verify constant values match Python

## Dependencies

- None (pure data module)

## Notes

- This module is data-only, no logic
- Can be generated from Python source
- Should be kept in sync with upstream PEFT
- Consider using a YAML/JSON file for easier updates
