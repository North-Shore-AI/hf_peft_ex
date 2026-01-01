# Mapping System Implementation

## Overview

The mapping system provides a registry pattern connecting PEFT method types to their configuration and tuner implementation classes.

## Python Reference

**File:** `peft/src/peft/mapping.py`

### Key Data Structures

```python
PEFT_TYPE_TO_CONFIG_MAPPING: dict[PeftType, type[PeftConfig]]
PEFT_TYPE_TO_TUNER_MAPPING: dict[PeftType, type[BaseTuner]]
PEFT_TYPE_TO_MIXED_MODEL_MAPPING: dict[PeftType, type]
PEFT_TYPE_TO_PREFIX_MAPPING: dict[PeftType, str]
```

### Key Functions

1. **`register_peft_method(name, config_cls, model_cls, prefix, is_mixed_compatible)`**
   - Registers a new PEFT method in all mappings
   - Validates unique name and prefix
   - Called during module import

2. **`get_peft_config(config_dict)`**
   - Factory function to instantiate appropriate config class from dict
   - Uses `peft_type` key to determine class

3. **`inject_adapter_in_model(peft_config, model, adapter_name)`**
   - Injects PEFT layers into a model in-place
   - Does NOT wrap model in PeftModel

## Elixir Implementation Design

### Module: `HfPeftEx.Mapping`

```elixir
defmodule HfPeftEx.Mapping do
  @moduledoc """
  Registry pattern for PEFT method type-to-implementation mappings.
  """

  # Type mappings stored as module attributes (compile-time)
  @peft_type_to_config %{
    lora: HfPeftEx.Tuners.Lora.Config,
    adalora: HfPeftEx.Tuners.Adalora.Config,
    ia3: HfPeftEx.Tuners.IA3.Config,
    # ... all 31 types
  }

  @peft_type_to_tuner %{
    lora: HfPeftEx.Tuners.Lora.Model,
    adalora: HfPeftEx.Tuners.Adalora.Model,
    ia3: HfPeftEx.Tuners.IA3.Model,
    # ... all 31 types
  }

  @peft_type_to_prefix %{
    lora: "lora_",
    adalora: "adalora_",
    ia3: "ia3_",
    # ... all 31 types
  }

  @spec get_config_class(atom()) :: module() | nil
  def get_config_class(peft_type)

  @spec get_tuner_class(atom()) :: module() | nil
  def get_tuner_class(peft_type)

  @spec get_prefix(atom()) :: String.t() | nil
  def get_prefix(peft_type)

  @spec get_peft_config(map()) :: {:ok, struct()} | {:error, term()}
  def get_peft_config(config_dict)

  @spec inject_adapter_in_model(struct(), map(), String.t()) :: {:ok, map()} | {:error, term()}
  def inject_adapter_in_model(peft_config, model, adapter_name \\ "default")
end
```

### Key Implementation Notes

1. Use module attributes for compile-time mappings (efficient)
2. Pattern match on atom types for O(1) lookup
3. `inject_adapter_in_model/3` will need Axon integration

## Files to Read

- `peft/src/peft/mapping.py` (full file)
- `peft/src/peft/utils/peft_types.py` (type definitions)
- `lib/hf_peft_ex/peft_type.ex` (existing Elixir types)

## Tests Required

1. Config class lookup for all 31 types
2. Tuner class lookup for all 31 types
3. Prefix lookup for all 31 types
4. `get_peft_config/1` factory function
5. Unknown type handling (returns nil or error)
6. `inject_adapter_in_model/3` with Axon models

## Dependencies

- `HfPeftEx.PeftType` (exists)
- All tuner config modules (as they're implemented)
- All tuner model modules (as they're implemented)
