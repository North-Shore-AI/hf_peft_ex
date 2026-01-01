# Tuners Utils Enhancement

## Overview

The tuners_utils module provides base abstractions for all tuner implementations. The Elixir port needs enhanced base behaviours and layer mixins.

## Python Reference

**File:** `peft/src/peft/tuners/tuners_utils.py` (~800 lines)

### Key Classes

#### 1. `BaseTuner` (Abstract Base Class)
```python
class BaseTuner(nn.Module, ABC):
    # Core attributes
    peft_config: dict[str, PeftConfig]
    active_adapter: str | list[str]
    active_adapters: list[str]  # property

    # Abstract methods (must implement)
    @abstractmethod
    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key)

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model)

    # Concrete methods
    def forward(self, *args, **kwargs)
    def merge_adapter(self, adapter_names)
    def unmerge_adapter(self)
    def inject_adapter(self, model, adapter_name, autocast_adapter_dtype)
    def set_adapter(self, adapter_name)
    def delete_adapter(self, adapter_name)
    def merge_and_unload(self, progressbar, safe_merge, adapter_names)
    def unload(self)
```

#### 2. `BaseTunerLayer` (Mixin)
```python
class BaseTunerLayer(ABC):
    # Common attributes
    r: dict[str, int]
    lora_alpha: dict[str, float]
    scaling: dict[str, float]
    active_adapter: str | list[str]
    merged_adapters: list[str]
    disable_adapters: bool
    _disable_adapters: bool
    merged: bool

    # Core methods
    @property
    def active_adapters(self) -> list[str]

    @property
    def merged(self) -> bool

    def merge(self, safe_merge, adapter_names)
    def unmerge(self)
    def enable_adapters(self, enabled)
    def set_adapter(self, adapter_names)
    def delete_adapter(self, adapter_name)
    def get_base_layer(self)
    def get_delta_weight(self, adapter)  # Abstract
```

#### 3. `LycorisTuner` (LyCORIS Base)
```python
class LycorisTuner(BaseTuner):
    # Shared implementation for LoHa and LoKr
    prefix: str

    def _create_and_replace(...)
    def _replace_module(...)
```

#### 4. `LycorisLayer` (LyCORIS Layer Mixin)
```python
class LycorisLayer(BaseTunerLayer):
    # Additional attributes for LyCORIS methods
    w1: dict
    w2: dict
    # Hadamard/Kronecker specific weights
```

### Target Module Matching

```python
def check_target_module_exists(config, key: str) -> bool
def _maybe_include_all_linear_layers(config, model) -> config
def _check_for_modules_to_save(config, model) -> config
def check_adapters_to_merge(module, adapter_names) -> list[str]
```

## Elixir Implementation Design

### Module: `HfPeftEx.Tuners.Base`

```elixir
defmodule HfPeftEx.Tuners.Base do
  @moduledoc """
  Base behaviour and shared functionality for all tuner implementations.
  """

  @callback create_and_replace(config :: struct(), adapter_name :: String.t(),
                               target :: map(), target_name :: String.t()) :: {:ok, map()} | {:error, term()}

  @callback mark_only_adapters_as_trainable(model :: map()) :: map()

  @callback get_prefix() :: String.t()

  # Shared functions
  @spec inject_adapter(model :: map(), config :: struct(), adapter_name :: String.t()) :: {:ok, map()} | {:error, term()}
  def inject_adapter(model, config, adapter_name \\ "default")

  @spec set_adapter(model :: map(), adapter_name :: String.t() | [String.t()]) :: {:ok, map()} | {:error, term()}
  def set_adapter(model, adapter_name)

  @spec delete_adapter(model :: map(), adapter_name :: String.t()) :: {:ok, map()} | {:error, term()}
  def delete_adapter(model, adapter_name)

  @spec merge_and_unload(model :: map(), opts :: keyword()) :: {:ok, map()} | {:error, term()}
  def merge_and_unload(model, opts \\ [])

  @spec check_target_module_exists(config :: struct(), key :: String.t()) :: boolean()
  def check_target_module_exists(config, key)
end
```

### Module: `HfPeftEx.Tuners.Layer`

```elixir
defmodule HfPeftEx.Tuners.Layer do
  @moduledoc """
  Base behaviour for all tuner layer implementations.
  """

  @callback merge(layer :: struct(), opts :: keyword()) :: {:ok, struct()} | {:error, term()}
  @callback unmerge(layer :: struct()) :: {:ok, struct()} | {:error, term()}
  @callback get_delta_weight(layer :: struct(), adapter :: String.t()) :: Nx.Tensor.t()
  @callback forward(layer :: struct(), input :: Nx.Tensor.t()) :: Nx.Tensor.t()

  # Shared struct fields (via __using__ macro)
  defmacro __using__(_opts) do
    quote do
      defstruct [
        :base_layer,
        :r,           # %{adapter_name => rank}
        :lora_alpha,  # %{adapter_name => alpha}
        :scaling,     # %{adapter_name => scaling}
        :active_adapter,
        :merged_adapters,
        :disable_adapters,
        merged: false
      ]
    end
  end
end
```

## Files to Read

- `peft/src/peft/tuners/tuners_utils.py` (full file)
- `lib/hf_peft_ex/tuners/lora/layer.ex` (current implementation)
- `lib/hf_peft_ex/tuners/lora/model.ex` (current implementation)

## Tests Required

1. `check_target_module_exists/2` with various patterns
2. Adapter injection into mock Axon model
3. Adapter switching (`set_adapter/2`)
4. Adapter deletion (`delete_adapter/2`)
5. Merge and unload operations
6. Multiple adapter handling

## Dependencies

- `HfPeftEx.PeftType`
- `HfPeftEx.Config`
- Existing LoRA layer implementation (for reference)
