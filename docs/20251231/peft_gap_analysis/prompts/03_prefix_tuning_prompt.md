# TDD Implementation Prompt: Prefix Tuning

## Task Overview

Implement the Prefix Tuning PEFT method for the `hf_peft_ex` Elixir library using Test-Driven Development.

Prefix Tuning prepends trainable prefix tokens to the keys and values of each attention layer. Unlike Prompt Tuning, it modifies all transformer layers.

---

## Required Reading

### Python Source Files (MUST READ FIRST)

1. **Configuration:** `peft/src/peft/tuners/prefix_tuning/config.py`
2. **Model & Encoder:** `peft/src/peft/tuners/prefix_tuning/model.py`
3. **Base Config:** `peft/src/peft/config.py` (PromptLearningConfig class)

### Elixir Context Files (READ FOR PATTERNS)

1. **Base Config:** `lib/hf_peft_ex/config.ex`
2. **PeftModel:** `lib/hf_peft_ex/peft_model.ex`

### Documentation Files

1. **Gap Analysis:** `docs/20251231/peft_gap_analysis/tuners/03_prefix_tuning.md`

---

## Implementation Requirements

### Files to Create

1. `lib/hf_peft_ex/tuners/prefix_tuning/config.ex`
2. `lib/hf_peft_ex/tuners/prefix_tuning/encoder.ex`
3. `lib/hf_peft_ex/tuners/prefix_tuning/model.ex`
4. `test/hf_peft_ex/tuners/prefix_tuning/config_test.exs`
5. `test/hf_peft_ex/tuners/prefix_tuning/encoder_test.exs`
6. `test/hf_peft_ex/tuners/prefix_tuning/model_test.exs`

---

## TDD Instructions

### Phase 1: Configuration

**Step 1: Write failing tests first**

```elixir
# test/hf_peft_ex/tuners/prefix_tuning/config_test.exs
defmodule HfPeftEx.Tuners.PrefixTuning.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :prefix_tuning
      assert config.num_virtual_tokens == 20
      assert config.prefix_projection == false
    end

    test "accepts prefix_projection" do
      config = Config.new(prefix_projection: true, encoder_hidden_size: 512)
      assert config.prefix_projection == true
      assert config.encoder_hidden_size == 512
    end

    test "validates projection requires encoder_hidden_size" do
      assert_raise ArgumentError, ~r/encoder_hidden_size required/, fn ->
        Config.new(prefix_projection: true)
      end
    end
  end

  describe "prefix_embedding_dim/1" do
    test "calculates total embedding dimension" do
      config = Config.new(
        num_layers: 12,
        token_dim: 768,
        num_virtual_tokens: 20
      )

      # num_layers * 2 (k+v) * token_dim
      assert Config.prefix_embedding_dim(config) == 12 * 2 * 768
    end
  end

  describe "is_prompt_learning/1" do
    test "returns true" do
      config = Config.new()
      assert Config.is_prompt_learning(config) == true
    end
  end
end
```

### Phase 2: Prefix Encoder

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/prefix_tuning/encoder_test.exs
defmodule HfPeftEx.Tuners.PrefixTuning.EncoderTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.{Config, Encoder}

  describe "new/1 without projection" do
    test "creates direct embedding" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: false
      )

      encoder = Encoder.new(config)

      total_dim = 4 * 2 * 64  # num_layers * 2 * token_dim
      assert Nx.shape(encoder.embedding) == {10, total_dim}
      assert encoder.use_projection == false
      assert encoder.transform == nil
    end
  end

  describe "new/1 with projection" do
    test "creates MLP projection" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: true,
        encoder_hidden_size: 128
      )

      encoder = Encoder.new(config)

      assert Nx.shape(encoder.embedding) == {10, 64}
      assert encoder.use_projection == true
      assert encoder.transform != nil
      assert Nx.shape(encoder.transform.w1) == {64, 128}
      assert Nx.shape(encoder.transform.w2) == {128, 4 * 2 * 64}
    end
  end

  describe "forward/2" do
    test "returns past_key_values with correct shape" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: false
      )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 2)

      # Shape: (batch, num_layers, 2, num_heads, num_tokens, head_dim)
      head_dim = div(64, 4)  # 16
      expected_shape = {2, 4, 2, 4, 10, head_dim}
      assert Nx.shape(output) == expected_shape
    end

    test "works with projection" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: true,
        encoder_hidden_size: 128
      )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 2)

      head_dim = div(64, 4)
      assert Nx.shape(output) == {2, 4, 2, 4, 10, head_dim}
    end
  end

  describe "get_trainable_params/1" do
    test "returns embedding for non-projection" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: false
      )

      encoder = Encoder.new(config)
      params = Encoder.get_trainable_params(encoder)

      assert Map.has_key?(params, "embedding")
      assert map_size(params) == 1
    end

    test "returns embedding and transform for projection" do
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4,
        prefix_projection: true,
        encoder_hidden_size: 128
      )

      encoder = Encoder.new(config)
      params = Encoder.get_trainable_params(encoder)

      assert Map.has_key?(params, "embedding")
      assert Map.has_key?(params, "transform.w1")
      assert Map.has_key?(params, "transform.b1")
      assert Map.has_key?(params, "transform.w2")
      assert Map.has_key?(params, "transform.b2")
    end
  end
end
```

### Phase 3: Model Integration

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/prefix_tuning/model_test.exs
defmodule HfPeftEx.Tuners.PrefixTuning.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.{Config, Model}

  describe "new/2" do
    test "creates model with prefix encoder" do
      base_model = create_mock_model()
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4
      )

      model = Model.new(base_model, config)

      assert model.config == config
      assert model.prefix_encoder != nil
    end

    test "auto-fills config from base model" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10)

      model = Model.new(base_model, config)

      assert model.config.num_layers == 4
      assert model.config.num_attention_heads == 4
      assert model.config.token_dim == 64
    end
  end

  describe "get_past_key_values/2" do
    test "generates past_key_values for batch" do
      base_model = create_mock_model()
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4
      )
      model = Model.new(base_model, config)

      past_key_values = Model.get_past_key_values(model, 2)

      # Should have entry for each layer
      assert length(past_key_values) == 4

      # Each entry is (key, value) tuple
      {key, value} = hd(past_key_values)
      assert Nx.shape(key) == {2, 4, 10, 16}  # batch, heads, seq, head_dim
      assert Nx.shape(value) == {2, 4, 10, 16}
    end
  end

  describe "prepare_attention_mask/3" do
    test "extends attention mask for prefix" do
      base_model = create_mock_model()
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4
      )
      model = Model.new(base_model, config)

      attention_mask = Nx.broadcast(1, {2, 20})
      new_mask = Model.prepare_attention_mask(model, 2, attention_mask)

      # Should be extended by num_virtual_tokens
      assert Nx.shape(new_mask) == {2, 30}
    end

    test "creates full attention mask if none provided" do
      base_model = create_mock_model()
      config = Config.new(
        num_virtual_tokens: 10,
        num_layers: 4,
        token_dim: 64,
        num_attention_heads: 4
      )
      model = Model.new(base_model, config)

      new_mask = Model.prepare_attention_mask(model, 2, nil)

      assert Nx.shape(new_mask) == {2, 10}
      assert Nx.to_number(Nx.sum(new_mask)) == 20  # All ones
    end
  end

  defp create_mock_model do
    %{
      config: %{
        num_hidden_layers: 4,
        num_attention_heads: 4,
        hidden_size: 64
      }
    }
  end
end
```

---

## Quality Requirements

### All Tests Must Pass

```bash
mix test test/hf_peft_ex/tuners/prefix_tuning/
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
## Prefix Tuning

Prefix Tuning prepends trainable prefix tokens to keys and values of each attention layer.

```elixir
config = HfPeftEx.Tuners.PrefixTuning.Config.new(
  num_virtual_tokens: 20,
  num_layers: 12,
  token_dim: 768,
  num_attention_heads: 12,
  prefix_projection: true,
  encoder_hidden_size: 512
)

model = HfPeftEx.Tuners.PrefixTuning.Model.new(base_model, config)

# Get past_key_values for attention layers
past_key_values = HfPeftEx.Tuners.PrefixTuning.Model.get_past_key_values(model, batch_size)
```

**Parameter count:**
- Without projection: `num_virtual_tokens * num_layers * 2 * token_dim`
- With projection: Much fewer due to bottleneck MLP
```

---

## Mathematical Reference

**Per-layer attention with prefix:**
```
K_l = [P_k^l, K_input]
V_l = [P_v^l, V_input]
Attention_l = softmax(Q @ K_l^T / sqrt(d)) @ V_l
```

**With projection (reparameterization):**
```
P = Tanh(E @ W1 + b1) @ W2 + b2
P_reshaped = reshape(P, (num_layers, 2, num_heads, num_tokens, head_dim))
```

---

## Completion Checklist

- [ ] All config tests pass
- [ ] All encoder tests pass
- [ ] All model tests pass
- [ ] No compiler warnings
- [ ] No dialyzer errors
- [ ] No credo issues
- [ ] Code formatted
- [ ] README.md updated with Prefix Tuning feature
- [ ] Documentation with @doc and @moduledoc
- [ ] Type specs (@spec) on all public functions
