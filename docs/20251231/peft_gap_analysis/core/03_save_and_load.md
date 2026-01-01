# Save and Load Utilities

## Overview

State dictionary operations for saving and loading adapter weights. Critical for model persistence and Hub integration.

## Python Reference

**File:** `peft/src/peft/utils/save_and_load.py` (724 lines)

### Key Functions

#### 1. `get_peft_model_state_dict(model, state_dict, adapter_name)`
**Purpose:** Extract only PEFT adapter weights from a model.

**Logic:**
1. Get base model state dict
2. Filter to only include adapter-related keys
3. Handle bias modes ("none", "all", "lora_only")
4. Special handling for:
   - DoRA magnitude vectors
   - VBLoRA top-k weights
   - VeRA projections
   - SHIRA indices
   - Trainable tokens
5. Remove adapter name prefix for portable state dict

**Returns:** `dict[str, Tensor]` - Adapter weights only

#### 2. `set_peft_model_state_dict(model, peft_model_state_dict, adapter_name)`
**Purpose:** Load PEFT weights into a model.

**Logic:**
1. Add adapter name prefix to keys
2. Handle auxiliary training wrappers
3. Handle mismatched sizes (optional ignore)
4. Type-specific restoration
5. Load into model

**Returns:** `IncompatibleKeys` - Missing/unexpected keys

#### 3. `load_peft_weights(model_id, device, **kwargs)`
**Purpose:** Load adapter weights from HuggingFace Hub or local path.

**Logic:**
1. Check for safetensors format (preferred)
2. Fall back to pickle (.bin) format
3. Handle offline mode
4. Apply key mapping if provided
5. Move tensors to device

**Returns:** `dict[str, Tensor]` - Loaded weights

### Helper Functions

```python
def _find_mismatched_keys(model_state_dict, loaded_state_dict) -> tuple[list, list]
def _insert_adapter_name_into_state_dict(state_dict, adapter_name) -> dict
def has_valid_embedding_base_layer(layer, embedding_adapter_name) -> bool
def get_embedding_layer_name(model, layer, adapter_name) -> Optional[str]
```

## Elixir Implementation Design

### Module: `HfPeftEx.Utils.SaveAndLoad`

```elixir
defmodule HfPeftEx.Utils.SaveAndLoad do
  @moduledoc """
  Utilities for saving and loading PEFT adapter state dictionaries.
  """

  @type state_dict :: %{String.t() => Nx.Tensor.t()}
  @type adapter_name :: String.t()

  @doc """
  Extract PEFT adapter weights from a model.

  Returns only the adapter-related weights, suitable for saving.
  Handles bias modes and special weight types.
  """
  @spec get_peft_model_state_dict(map(), adapter_name(), keyword()) :: {:ok, state_dict()} | {:error, term()}
  def get_peft_model_state_dict(model, adapter_name \\ "default", opts \\ [])

  @doc """
  Load PEFT weights into a model.

  Handles key mapping, adapter name prefixing, and shape validation.
  """
  @spec set_peft_model_state_dict(map(), state_dict(), adapter_name()) :: {:ok, map()} | {:error, term()}
  def set_peft_model_state_dict(model, state_dict, adapter_name \\ "default")

  @doc """
  Load adapter weights from file path.

  Supports safetensors format (preferred) and Nx serialization.
  """
  @spec load_peft_weights(String.t(), keyword()) :: {:ok, state_dict()} | {:error, term()}
  def load_peft_weights(path, opts \\ [])

  @doc """
  Save adapter weights to file path.

  Uses safetensors format by default.
  """
  @spec save_peft_weights(state_dict(), String.t(), keyword()) :: :ok | {:error, term()}
  def save_peft_weights(state_dict, path, opts \\ [])

  # Private helpers
  defp filter_adapter_keys(state_dict, adapter_name, config)
  defp add_adapter_prefix(state_dict, adapter_name)
  defp remove_adapter_prefix(state_dict, adapter_name)
  defp handle_bias_mode(state_dict, bias_mode)
end
```

### File Format Considerations

**Elixir/Nx Options:**
1. **Safetensors** - Cross-platform, compatible with Python
   - Use `safetensors` library (Rust-based, has Elixir bindings potential)
2. **Nx.serialize/deserialize** - Native Nx format
   - Fast, but Elixir-only
3. **JSON + Base64** - Portable but inefficient
   - Good for small adapters

**Recommendation:** Implement safetensors first for HuggingFace Hub compatibility.

### Key Prefix Patterns

```elixir
# LoRA keys
"base_model.model.{layer_name}.lora_A.{adapter_name}.weight"
"base_model.model.{layer_name}.lora_B.{adapter_name}.weight"

# After extraction (portable format)
"{layer_name}.lora_A.weight"
"{layer_name}.lora_B.weight"
```

## Files to Read

- `peft/src/peft/utils/save_and_load.py` (full file)
- `lib/hf_peft_ex/peft_model.ex` (current save/load stubs)
- `lib/hf_peft_ex/config.ex` (JSON serialization)

## Tests Required

1. Extract state dict from LoRA model
2. Filter by adapter name
3. Handle bias modes (none, all, lora_only)
4. Save to safetensors format
5. Load from safetensors format
6. Round-trip test (save + load = identical)
7. Handle missing/unexpected keys gracefully
8. DoRA magnitude vector handling
9. Multi-adapter state dicts

## Dependencies

- `HfPeftEx.Config`
- `HfPeftEx.Mapping` (for type detection)
- Safetensors library (or Nx serialization)
- `Jason` for config JSON

## Integration Points

- `HfPeftEx.PeftModel.save_pretrained/2`
- `HfPeftEx.PeftModel.from_pretrained/2`
- HuggingFace Hub (future)
