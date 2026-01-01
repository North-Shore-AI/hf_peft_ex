# TDD Implementation Prompt: IA3 (Infused Adapter by Inhibiting and Amplifying Activations)

## Task Overview

Implement the IA3 PEFT method for the `hf_peft_ex` Elixir library using Test-Driven Development.

IA3 learns multiplicative scaling vectors instead of low-rank matrices. It's simpler than LoRA and has minimal parameter overhead.

---

## Required Reading

### Python Source Files (MUST READ FIRST)

1. **Configuration:** `peft/src/peft/tuners/ia3/config.py`
2. **Layer Implementation:** `peft/src/peft/tuners/ia3/layer.py`
3. **Model Wrapper:** `peft/src/peft/tuners/ia3/model.py`
4. **Init Exports:** `peft/src/peft/tuners/ia3/__init__.py`

### Elixir Context Files (READ FOR PATTERNS)

1. **LoRA Config Pattern:** `lib/hf_peft_ex/tuners/lora/config.ex`
2. **LoRA Layer Pattern:** `lib/hf_peft_ex/tuners/lora/layer.ex`
3. **LoRA Linear Pattern:** `lib/hf_peft_ex/tuners/lora/linear.ex`
4. **Base Config:** `lib/hf_peft_ex/config.ex`
5. **PeftType Enum:** `lib/hf_peft_ex/peft_type.ex`

### Documentation Files

1. **Gap Analysis:** `docs/20251231/peft_gap_analysis/tuners/01_ia3.md`

---

## Implementation Requirements

### Files to Create

1. `lib/hf_peft_ex/tuners/ia3/config.ex`
2. `lib/hf_peft_ex/tuners/ia3/layer.ex`
3. `lib/hf_peft_ex/tuners/ia3/linear.ex`
4. `lib/hf_peft_ex/tuners/ia3/model.ex`
5. `test/hf_peft_ex/tuners/ia3/config_test.exs`
6. `test/hf_peft_ex/tuners/ia3/layer_test.exs`
7. `test/hf_peft_ex/tuners/ia3/linear_test.exs`
8. `test/hf_peft_ex/tuners/ia3/model_test.exs`

---

## TDD Instructions

### Phase 1: Configuration (IA3Config)

**Step 1: Write failing tests first**

```elixir
# test/hf_peft_ex/tuners/ia3/config_test.exs
defmodule HfPeftEx.Tuners.IA3.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :ia3
      assert config.init_ia3_weights == true
      assert config.fan_in_fan_out == false
    end

    test "accepts target_modules as list" do
      config = Config.new(target_modules: ["q_proj", "v_proj"])
      assert config.target_modules == ["q_proj", "v_proj"]
    end

    test "accepts feedforward_modules" do
      config = Config.new(
        target_modules: ["k_proj", "v_proj", "down_proj"],
        feedforward_modules: ["down_proj"]
      )
      assert config.feedforward_modules == ["down_proj"]
    end

    test "validates feedforward_modules is subset of target_modules" do
      assert_raise ArgumentError, fn ->
        Config.new(
          target_modules: ["q_proj", "v_proj"],
          feedforward_modules: ["down_proj"]
        )
      end
    end
  end

  describe "JSON serialization" do
    test "to_json/1 and from_json/1 round trip" do
      config = Config.new(target_modules: ["q_proj"], init_ia3_weights: false)
      json = Config.to_json(config)
      decoded = Config.from_json(json)
      assert decoded.target_modules == config.target_modules
      assert decoded.init_ia3_weights == config.init_ia3_weights
    end
  end
end
```

**Step 2: Implement to make tests pass**

### Phase 2: Layer Implementation

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/ia3/layer_test.exs
defmodule HfPeftEx.Tuners.IA3.LayerTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.Layer

  describe "new/3" do
    test "creates layer with correct dimensions" do
      base_layer = %{weight: Nx.random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.in_features == 32
      assert layer.out_features == 64
      assert layer.is_feedforward == false
    end

    test "handles fan_in_fan_out" do
      base_layer = %{weight: Nx.random_normal({32, 64})}
      layer = Layer.new(base_layer, false, fan_in_fan_out: true)

      assert layer.in_features == 32
      assert layer.out_features == 64
    end
  end

  describe "update_layer/3" do
    test "initializes ia3_l to ones when init_ia3_weights is true" do
      base_layer = %{weight: Nx.random_normal({64, 32})}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      ia3_l = layer.ia3_l["default"]
      assert Nx.shape(ia3_l) == {64}
      assert Nx.to_number(Nx.mean(ia3_l)) == 1.0
    end

    test "uses out_features for non-feedforward" do
      base_layer = %{weight: Nx.random_normal({64, 32})}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      assert Nx.shape(layer.ia3_l["default"]) == {64}
    end

    test "uses in_features for feedforward" do
      base_layer = %{weight: Nx.random_normal({64, 32})}
      layer = Layer.new(base_layer, true)
      |> Layer.update_layer("default", true)

      assert Nx.shape(layer.ia3_l["default"]) == {32}
    end
  end

  describe "forward/2" do
    test "returns identity when ia3_l is ones" do
      base_layer = %{weight: Nx.eye(4), bias: nil}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      assert_all_close(output, input)
    end

    test "scales output by ia3_l" do
      base_layer = %{weight: Nx.eye(4), bias: nil}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      # Manually set scaling
      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0]])
      assert_all_close(output, expected)
    end

    test "returns base output when adapters disabled" do
      base_layer = %{weight: Nx.eye(4), bias: nil}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)
      |> Map.put(:disable_adapters, true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      assert_all_close(output, input)
    end
  end

  describe "merge/2" do
    test "modifies base weights correctly for non-feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))
      {:ok, merged} = Layer.merge(layer)

      # Rows should be scaled
      expected = Nx.tensor([
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0]
      ])
      assert_all_close(merged.base_layer.weight, expected)
      assert merged.merged == true
    end
  end

  describe "unmerge/1" do
    test "restores original weights" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}
      layer = Layer.new(base_layer, false)
      |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))
      {:ok, merged} = Layer.merge(layer)
      {:ok, unmerged} = Layer.unmerge(merged)

      assert_all_close(unmerged.base_layer.weight, Nx.broadcast(1.0, {4, 4}))
      assert unmerged.merged == false
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
```

**Step 2: Implement layer to pass tests**

### Phase 3: Model Implementation

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/ia3/model_test.exs
defmodule HfPeftEx.Tuners.IA3.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.{Config, Model}

  describe "new/2" do
    test "wraps base model with IA3 layers" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1", "layer2"])

      model = Model.new(base_model, config)

      assert model.config == config
      assert Map.has_key?(model.ia3_layers, "layer1")
      assert Map.has_key?(model.ia3_layers, "layer2")
    end

    test "identifies feedforward modules correctly" do
      base_model = create_mock_model()
      config = Config.new(
        target_modules: ["layer1", "layer2"],
        feedforward_modules: ["layer2"]
      )

      model = Model.new(base_model, config)

      refute model.ia3_layers["layer1"].is_feedforward
      assert model.ia3_layers["layer2"].is_feedforward
    end
  end

  describe "set_adapter/2" do
    test "switches active adapter" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)
      |> Model.add_adapter("adapter2")
      |> Model.set_adapter("adapter2")

      assert model.active_adapter == "adapter2"
    end
  end

  describe "get_trainable_params/1" do
    test "returns only ia3_l parameters" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1", "layer2"])
      model = Model.new(base_model, config)

      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "layer1.ia3_l.default")
      assert Map.has_key?(params, "layer2.ia3_l.default")
      assert map_size(params) == 2
    end
  end

  defp create_mock_model do
    %{
      layer1: %{weight: Nx.random_normal({64, 32}), bias: nil},
      layer2: %{weight: Nx.random_normal({64, 64}), bias: nil}
    }
  end
end
```

---

## Quality Requirements

### All Tests Must Pass

```bash
mix test test/hf_peft_ex/tuners/ia3/
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
## Implemented PEFT Methods

### IA3 (Infused Adapter by Inhibiting and Amplifying Activations)

IA3 learns multiplicative scaling vectors for layer activations. Much simpler than LoRA with minimal parameters.

```elixir
config = HfPeftEx.Tuners.IA3.Config.new(
  target_modules: ["q_proj", "v_proj", "down_proj"],
  feedforward_modules: ["down_proj"],
  init_ia3_weights: true
)

# Create IA3 layer
layer = HfPeftEx.Tuners.IA3.Layer.new(base_layer, is_feedforward)
|> HfPeftEx.Tuners.IA3.Layer.update_layer("default", true)

# Forward pass
output = HfPeftEx.Tuners.IA3.Layer.forward(layer, input)
```

**Parameter count:** Only `d` parameters per layer (the scaling vector).
```

---

## Mathematical Reference

**IA3 Forward:**
```
y = (Wx) * l
```
Where `l` is the learned scaling vector.

**Merge:**
```
W' = W * diag(l)
```

**Unmerge:**
```
W = W' / diag(l)
```

---

## Completion Checklist

- [ ] All config tests pass
- [ ] All layer tests pass
- [ ] All linear tests pass
- [ ] All model tests pass
- [ ] No compiler warnings
- [ ] No dialyzer errors
- [ ] No credo issues
- [ ] Code formatted
- [ ] README.md updated with IA3 feature
- [ ] Documentation with @doc and @moduledoc
- [ ] Type specs (@spec) on all public functions
