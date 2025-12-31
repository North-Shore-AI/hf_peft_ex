# HF PEFT Elixir Port - Continuation Prompt

You are continuing the development of `hf_peft_ex`, an Elixir port of HuggingFace's PEFT (Parameter-Efficient Fine-Tuning) library.

## Current State

The following core modules have been implemented:

```
lib/
├── hf_peft_ex.ex                           # Main module
├── hf_peft_ex/
│   ├── peft_type.ex                        # 31 PEFT method types (atoms)
│   ├── task_type.ex                        # 6 task types
│   ├── config.ex                           # Base config with JSON serialization
│   └── tuners/
│       └── lora/
│           └── config.ex                   # LoRA configuration (20+ options)
```

**Tests:** 19 tests pass (`mix test`)

**GitHub:** https://github.com/North-Shore-AI/hf_peft_ex

**Original Python source:** Located in `./peft/` directory

---

## Your Task: Continue Porting with TDD

Use **Test-Driven Development** for all new code:
1. Write failing tests first
2. Implement code to make tests pass
3. Refactor while keeping tests green

### Priority Order

#### 1. LoRA Layer (`lib/hf_peft_ex/tuners/lora/layer.ex`)

The LoRA layer wraps a linear layer and injects low-rank adapters.

**Key functionality to port from `peft/src/peft/tuners/lora/layer.py`:**

```elixir
# Write tests first for:
defmodule HfPeftEx.Tuners.Lora.LayerTest do
  # Test LoRA layer initialization
  # Test forward pass with scaling: output = base_output + (x @ A^T @ B^T) * scaling
  # Test merged/unmerged states
  # Test dropout application
  # Test DoRA magnitude calculation
end
```

**Required struct fields:**
- `base_layer` - The original layer being wrapped
- `r` - Rank
- `lora_alpha` - Scaling factor
- `lora_A` - Low-rank matrix A (r × in_features)
- `lora_B` - Low-rank matrix B (out_features × r)
- `scaling` - Computed scaling factor
- `merged` - Whether weights are merged

---

#### 2. LoRA Model (`lib/hf_peft_ex/tuners/lora/model.ex`)

The model wrapper that applies LoRA to a base model.

**Key functionality to port from `peft/src/peft/tuners/lora/model.py`:**

```elixir
# Write tests first for:
defmodule HfPeftEx.Tuners.Lora.ModelTest do
  # Test wrapping a model with LoRA config
  # Test target_modules matching (exact, regex, "all-linear")
  # Test exclude_modules filtering
  # Test adapter enable/disable
  # Test merge/unmerge operations
  # Test saving/loading adapters
end
```

---

#### 3. PeftModel (`lib/hf_peft_ex/peft_model.ex`)

Generic PEFT model wrapper.

```elixir
# Write tests first for:
defmodule HfPeftEx.PeftModelTest do
  # Test creating PEFT model from config
  # Test get_peft_config
  # Test print_trainable_parameters
  # Test save_pretrained
  # Test from_pretrained
end
```

---

## Testing Requirements

For each module:

1. **Create test file first** in `test/hf_peft_ex/tuners/lora/`
2. **Run tests to see them fail:** `mix test`
3. **Implement minimum code to pass**
4. **Run tests to verify:** `mix test`
5. **Refactor if needed**

### Example TDD Cycle

```bash
# Step 1: Write failing test
# test/hf_peft_ex/tuners/lora/layer_test.exs
defmodule HfPeftEx.Tuners.Lora.LayerTest do
  use ExUnit.Case
  alias HfPeftEx.Tuners.Lora.Layer

  test "create initializes with correct dimensions" do
    layer = Layer.new(in_features: 1024, out_features: 1024, r: 8)
    assert layer.r == 8
    # lora_A should be r × in_features
    assert Nx.shape(layer.lora_A) == {8, 1024}
    # lora_B should be out_features × r
    assert Nx.shape(layer.lora_B) == {1024, 8}
  end
end

# Step 2: Run test, see it fail
$ mix test test/hf_peft_ex/tuners/lora/layer_test.exs
# Expected: 1 failure

# Step 3: Implement Layer.new/1 to make it pass
# lib/hf_peft_ex/tuners/lora/layer.ex

# Step 4: Run test, see it pass
$ mix test test/hf_peft_ex/tuners/lora/layer_test.exs
# Expected: 1 test, 0 failures
```

---

## Dependencies to Add

Update `mix.exs`:

```elixir
defp deps do
  [
    {:jason, "~> 1.4"},
    {:nx, "~> 0.7"},           # Numerical computing
    {:axon, "~> 0.6"},         # Neural networks
    {:exla, "~> 0.7"}          # Optional: GPU acceleration
  ]
end
```

---

## Key Python Files to Reference

| Elixir Module | Python Source |
|---------------|---------------|
| `Lora.Layer` | `peft/src/peft/tuners/lora/layer.py` |
| `Lora.Model` | `peft/src/peft/tuners/lora/model.py` |
| `PeftModel` | `peft/src/peft/peft_model.py` |
| `tuners_utils` | `peft/src/peft/tuners/tuners_utils.py` |

---

## Quality Checks

After each module:

```bash
# All tests pass
mix test

# Code quality
mix credo --strict

# Type checking (optional)
mix dialyzer
```

---

## Commit Guidelines

Commit after each completed module:

```bash
git add . && git commit -m "Add HfPeftEx.Tuners.Lora.Layer with TDD

- Add Layer struct with lora_A, lora_B matrices
- Implement forward pass with scaling
- Implement merge/unmerge operations
- Add 12 tests covering initialization, forward, and merge"
```

Then push: `git push origin main`
