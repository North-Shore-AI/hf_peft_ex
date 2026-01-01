# TDD Implementation Prompt: AdaLoRA (Adaptive LoRA)

## Task Overview

Implement the AdaLoRA PEFT method for the `hf_peft_ex` Elixir library using Test-Driven Development.

AdaLoRA dynamically allocates rank budget across layers during training based on importance scores. Less important layers get pruned while important layers retain higher rank.

---

## Required Reading

### Python Source Files (MUST READ FIRST)

1. **Configuration:** `peft/src/peft/tuners/adalora/config.py`
2. **Layer Implementation:** `peft/src/peft/tuners/adalora/layer.py`
3. **Model & RankAllocator:** `peft/src/peft/tuners/adalora/model.py`

### Elixir Context Files (READ FOR PATTERNS)

1. **LoRA Config:** `lib/hf_peft_ex/tuners/lora/config.ex`
2. **LoRA Layer:** `lib/hf_peft_ex/tuners/lora/layer.ex`
3. **LoRA Linear:** `lib/hf_peft_ex/tuners/lora/linear.ex`

### Documentation Files

1. **Gap Analysis:** `docs/20251231/peft_gap_analysis/tuners/04_adalora.md`

---

## Implementation Requirements

### Files to Create

1. `lib/hf_peft_ex/tuners/adalora/config.ex`
2. `lib/hf_peft_ex/tuners/adalora/layer.ex`
3. `lib/hf_peft_ex/tuners/adalora/rank_allocator.ex`
4. `lib/hf_peft_ex/tuners/adalora/model.ex`
5. `test/hf_peft_ex/tuners/adalora/config_test.exs`
6. `test/hf_peft_ex/tuners/adalora/layer_test.exs`
7. `test/hf_peft_ex/tuners/adalora/rank_allocator_test.exs`
8. `test/hf_peft_ex/tuners/adalora/model_test.exs`

---

## TDD Instructions

### Phase 1: Configuration

**Step 1: Write failing tests first**

```elixir
# test/hf_peft_ex/tuners/adalora/config_test.exs
defmodule HfPeftEx.Tuners.Adalora.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new(total_step: 1000)
      assert config.peft_type == :adalora
      assert config.init_r == 12
      assert config.target_r == 8
      assert config.tinit == 0
      assert config.tfinal == 0
      assert config.delta_t == 1
      assert config.beta1 == 0.85
      assert config.beta2 == 0.85
      assert config.orth_reg_weight == 0.5
    end

    test "validates total_step is required" do
      assert_raise ArgumentError, ~r/total_step is required/, fn ->
        Config.new()
      end
    end

    test "validates total_step > 0" do
      assert_raise ArgumentError, fn ->
        Config.new(total_step: 0)
      end
    end

    test "validates tinit < total_step - tfinal" do
      assert_raise ArgumentError, fn ->
        Config.new(total_step: 100, tinit: 80, tfinal: 30)
      end
    end

    test "validates init_r >= target_r" do
      assert_raise ArgumentError, fn ->
        Config.new(total_step: 100, init_r: 4, target_r: 8)
      end
    end
  end

  describe "inherits LoRA config fields" do
    test "has lora_alpha and lora_dropout" do
      config = Config.new(total_step: 100, lora_alpha: 32, lora_dropout: 0.1)
      assert config.lora_alpha == 32
      assert config.lora_dropout == 0.1
    end

    test "has target_modules" do
      config = Config.new(total_step: 100, target_modules: ["q_proj", "v_proj"])
      assert config.target_modules == ["q_proj", "v_proj"]
    end
  end
end
```

### Phase 2: Layer Implementation (SVD Parameterization)

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/adalora/layer_test.exs
defmodule HfPeftEx.Tuners.Adalora.LayerTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.Layer

  describe "new/2" do
    test "creates layer with correct dimensions" do
      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)

      assert layer.in_features == 32
      assert layer.out_features == 64
    end
  end

  describe "update_layer/4" do
    test "initializes SVD parameterization" do
      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 8, 16)

      # lora_a is Q (r x in_features)
      assert Nx.shape(layer.lora_a["default"]) == {8, 32}
      # lora_b is P (out_features x r)
      assert Nx.shape(layer.lora_b["default"]) == {64, 8}
      # lora_e is Lambda (r x 1)
      assert Nx.shape(layer.lora_e["default"]) == {8, 1}
      # Initial rank
      assert layer.ranknum["default"] == 8
    end

    test "initializes lora_e to ones" do
      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 8, 16)

      assert Nx.to_number(Nx.mean(layer.lora_e["default"])) == 1.0
    end
  end

  describe "forward/2" do
    test "computes delta weight correctly" do
      base_layer = %{weight: Nx.eye(8), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 4, 4)

      # Set known values for predictable output
      layer = layer
      |> put_in([:lora_a, "default"], Nx.eye(4) |> Nx.pad([[0, 0], [0, 4]], 0.0))  # 4x8
      |> put_in([:lora_b, "default"], Nx.eye(4) |> Nx.pad([[0, 4], [0, 0]], 0.0))  # 8x4
      |> put_in([:lora_e, "default"], Nx.broadcast(1.0, {4, 1}))
      |> put_in([:scaling, "default"], 1.0)

      input = Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      output = Layer.forward(layer, input)

      # Base output + delta
      # Delta = x @ (E * A)^T @ B^T
      assert Nx.shape(output) == {1, 8}
    end

    test "returns base output when disabled" do
      base_layer = %{weight: Nx.eye(4), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 2, 2)
      |> Map.put(:disable_adapters, true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      assert_all_close(output, input)
    end
  end

  describe "get_delta_weight/2" do
    test "computes B @ diag(E) @ A" do
      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 8, 16)

      delta = Layer.get_delta_weight(layer, "default")

      assert Nx.shape(delta) == {64, 32}
    end
  end

  describe "apply_mask/3" do
    test "zeros out pruned singular values" do
      base_layer = %{weight: Nx.random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)
      |> Layer.update_layer("default", 8, 16)

      # Mask: keep first 4, prune last 4
      mask = Nx.concatenate([Nx.broadcast(1.0, {4, 1}), Nx.broadcast(0.0, {4, 1})], axis: 0)
      masked_layer = Layer.apply_mask(layer, "default", mask)

      # Check rank updated
      assert masked_layer.ranknum["default"] == 4

      # Check E is masked
      e = masked_layer.lora_e["default"]
      assert Nx.to_number(e[4][0]) == 0.0
      assert Nx.to_number(e[5][0]) == 0.0
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
```

### Phase 3: Rank Allocator

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/adalora/rank_allocator_test.exs
defmodule HfPeftEx.Tuners.Adalora.RankAllocatorTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.{Config, RankAllocator}

  describe "new/2" do
    test "creates allocator with config" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config)

      assert allocator.config == config
      assert allocator.ipt == %{}
      assert allocator.exp_avg_ipt == %{}
    end
  end

  describe "compute_importance/2" do
    test "computes |param * gradient|" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config)

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[1.0], [2.0], [3.0]]),
          lora_e_grad: Nx.tensor([[0.1], [0.2], [0.3]])
        }
      }

      updated = RankAllocator.compute_importance(allocator, gradients)

      expected = Nx.tensor([[0.1], [0.4], [0.9]])
      assert_all_close(updated.ipt["layer1"], expected)
    end
  end

  describe "update_ema/1" do
    test "applies exponential moving average" do
      config = Config.new(total_step: 1000, beta1: 0.5)
      allocator = RankAllocator.new(config)

      # First importance
      allocator = %{allocator |
        ipt: %{"layer1" => Nx.tensor([[1.0], [2.0]])}
      }
      allocator = RankAllocator.update_ema(allocator)

      # EMA = 0.5 * 0 + 0.5 * [1, 2] = [0.5, 1.0]
      expected = Nx.tensor([[0.5], [1.0]])
      assert_all_close(allocator.exp_avg_ipt["layer1"], expected)

      # Second importance
      allocator = %{allocator |
        ipt: %{"layer1" => Nx.tensor([[2.0], [4.0]])}
      }
      allocator = RankAllocator.update_ema(allocator)

      # EMA = 0.5 * [0.5, 1.0] + 0.5 * [2, 4] = [1.25, 2.5]
      expected = Nx.tensor([[1.25], [2.5]])
      assert_all_close(allocator.exp_avg_ipt["layer1"], expected)
    end
  end

  describe "get_budget/2" do
    test "returns init_r during warmup" do
      config = Config.new(total_step: 1000, tinit: 100, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      assert RankAllocator.get_budget(allocator, 50) == 12.0
    end

    test "linearly decreases budget" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      # At step 500 (50% progress), budget should be midpoint
      budget = RankAllocator.get_budget(allocator, 500)
      assert_in_delta budget, 10.0, 0.1
    end

    test "returns target_r at end" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      budget = RankAllocator.get_budget(allocator, 1000)
      assert budget == 8.0
    end
  end

  describe "compute_masks/2" do
    test "creates masks based on importance threshold" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)
      allocator = %{RankAllocator.new(config) |
        exp_avg_ipt: %{
          "layer1" => Nx.tensor([[1.0], [0.5], [0.2], [0.1]]),
          "layer2" => Nx.tensor([[0.8], [0.3], [0.15], [0.05]])
        },
        exp_avg_unc: %{
          "layer1" => Nx.tensor([[0.0], [0.0], [0.0], [0.0]]),
          "layer2" => Nx.tensor([[0.0], [0.0], [0.0], [0.0]])
        }
      }

      # Budget = 2 per layer (target_r)
      masks = RankAllocator.compute_masks(allocator, 2.0)

      # Should keep top 50% by importance
      assert Nx.to_number(Nx.sum(masks["layer1"])) == 2
      assert Nx.to_number(Nx.sum(masks["layer2"])) == 2
    end
  end

  describe "update_and_allocate/3" do
    test "returns nil during warmup" do
      config = Config.new(total_step: 1000, tinit: 100)
      allocator = RankAllocator.new(config)

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, %{}, 50)
      assert masks == nil
    end

    test "returns nil during final phase" do
      config = Config.new(total_step: 1000, tfinal: 100)
      allocator = RankAllocator.new(config)

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, %{}, 950)
      assert masks == nil
    end

    test "returns masks at pruning intervals" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10)
      allocator = %{RankAllocator.new(config) |
        exp_avg_ipt: %{"layer1" => Nx.tensor([[1.0], [0.5]])},
        exp_avg_unc: %{"layer1" => Nx.tensor([[0.0], [0.0]])}
      }

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[1.0], [1.0]]),
          lora_e_grad: Nx.tensor([[0.1], [0.2]])
        }
      }

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, gradients, 100)
      assert masks != nil
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
```

---

## Quality Requirements

### All Tests Must Pass

```bash
mix test test/hf_peft_ex/tuners/adalora/
```

### No Warnings

```bash
mix compile --warnings-as-errors
```

### No Dialyzer Errors

```bash
mix dialyzer
```

### No Credo Issues

```bash
mix credo --strict
```

### Format Code

```bash
mix format
```

---

## README Update

After implementation, add to README.md:

```markdown
## AdaLoRA (Adaptive LoRA)

AdaLoRA dynamically allocates rank budget across layers based on importance scores during training.

```elixir
config = HfPeftEx.Tuners.Adalora.Config.new(
  total_step: 10000,
  init_r: 12,
  target_r: 8,
  tinit: 200,
  tfinal: 200,
  delta_t: 10,
  beta1: 0.85,
  target_modules: ["q_proj", "v_proj"]
)

# Layer uses SVD parameterization: W = W0 + P @ diag(Lambda) @ Q^T
layer = HfPeftEx.Tuners.Adalora.Layer.new(base_layer)
|> HfPeftEx.Tuners.Adalora.Layer.update_layer("default", init_r, lora_alpha)

# Rank allocator manages pruning during training
allocator = HfPeftEx.Tuners.Adalora.RankAllocator.new(config)
{allocator, masks} = HfPeftEx.Tuners.Adalora.RankAllocator.update_and_allocate(
  allocator, gradients, step
)
```

**Key features:**
- SVD parameterization with prunable singular values
- Dynamic rank allocation based on importance
- EMA smoothing of importance scores
```

---

## Mathematical Reference

**SVD Parameterization:**
```
Delta W = P @ diag(Lambda) @ Q^T
```

**Importance Score:**
```
I(lambda_i) = |lambda_i * grad(lambda_i)|
```

**Budget Schedule:**
```
budget(t) = init_r - progress * (init_r - target_r)
where progress = (t - tinit) / (total - tfinal - tinit)
```

---

## Completion Checklist

- [ ] All config tests pass
- [ ] All layer tests pass
- [ ] All rank_allocator tests pass
- [ ] All model tests pass
- [ ] No compiler warnings
- [ ] No dialyzer errors
- [ ] No credo issues
- [ ] Code formatted
- [ ] README.md updated with AdaLoRA feature
- [ ] Documentation with @doc and @moduledoc
- [ ] Type specs (@spec) on all public functions
