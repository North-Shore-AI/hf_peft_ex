# TDD Implementation Prompt: Prompt Tuning

## Task Overview

Implement the Prompt Tuning PEFT method for the `hf_peft_ex` Elixir library using Test-Driven Development.

Prompt Tuning adds learnable virtual tokens (soft prompts) to input embeddings. These virtual tokens are prepended to the input and learned during training.

---

## Required Reading

### Python Source Files (MUST READ FIRST)

1. **Configuration:** `peft/src/peft/tuners/prompt_tuning/config.py`
2. **Model & Embedding:** `peft/src/peft/tuners/prompt_tuning/model.py`
3. **Base Config:** `peft/src/peft/config.py` (PromptLearningConfig class)

### Elixir Context Files (READ FOR PATTERNS)

1. **Base Config:** `lib/hf_peft_ex/config.ex`
2. **PeftModel:** `lib/hf_peft_ex/peft_model.ex`
3. **LoRA Config Pattern:** `lib/hf_peft_ex/tuners/lora/config.ex`

### Documentation Files

1. **Gap Analysis:** `docs/20251231/peft_gap_analysis/tuners/02_prompt_tuning.md`

---

## Implementation Requirements

### Files to Create

1. `lib/hf_peft_ex/tuners/prompt_tuning/config.ex`
2. `lib/hf_peft_ex/tuners/prompt_tuning/embedding.ex`
3. `lib/hf_peft_ex/tuners/prompt_tuning/model.ex`
4. `test/hf_peft_ex/tuners/prompt_tuning/config_test.exs`
5. `test/hf_peft_ex/tuners/prompt_tuning/embedding_test.exs`
6. `test/hf_peft_ex/tuners/prompt_tuning/model_test.exs`

---

## TDD Instructions

### Phase 1: Configuration

**Step 1: Write failing tests first**

```elixir
# test/hf_peft_ex/tuners/prompt_tuning/config_test.exs
defmodule HfPeftEx.Tuners.PromptTuning.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :prompt_tuning
      assert config.num_virtual_tokens == 20
      assert config.prompt_tuning_init == :random
    end

    test "accepts num_virtual_tokens" do
      config = Config.new(num_virtual_tokens: 50)
      assert config.num_virtual_tokens == 50
    end

    test "accepts prompt_tuning_init as atom" do
      config = Config.new(prompt_tuning_init: :text)
      assert config.prompt_tuning_init == :text
    end

    test "validates text init requires init_text" do
      assert_raise ArgumentError, ~r/prompt_tuning_init_text required/, fn ->
        Config.new(prompt_tuning_init: :text)
      end
    end

    test "validates text init requires tokenizer" do
      assert_raise ArgumentError, ~r/tokenizer_name_or_path required/, fn ->
        Config.new(
          prompt_tuning_init: :text,
          prompt_tuning_init_text: "Classify this text:"
        )
      end
    end

    test "validates num_virtual_tokens > 0" do
      assert_raise ArgumentError, fn ->
        Config.new(num_virtual_tokens: 0)
      end
    end
  end

  describe "is_prompt_learning/1" do
    test "returns true" do
      config = Config.new()
      assert Config.is_prompt_learning(config) == true
    end
  end

  describe "JSON serialization" do
    test "round trip preserves config" do
      config = Config.new(
        num_virtual_tokens: 30,
        prompt_tuning_init: :random,
        token_dim: 768
      )
      json = Config.to_json(config)
      decoded = Config.from_json(json)

      assert decoded.num_virtual_tokens == 30
      assert decoded.prompt_tuning_init == :random
      assert decoded.token_dim == 768
    end
  end
end
```

### Phase 2: Prompt Embedding

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/prompt_tuning/embedding_test.exs
defmodule HfPeftEx.Tuners.PromptTuning.EmbeddingTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.{Config, Embedding}

  describe "new/2 with random init" do
    test "creates embedding with correct shape" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      assert Nx.shape(embedding.embedding) == {10, 64}
      assert embedding.num_virtual_tokens == 10
      assert embedding.token_dim == 64
    end

    test "initializes with small random values" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      mean = Nx.mean(embedding.embedding) |> Nx.to_number()
      std = Nx.standard_deviation(embedding.embedding) |> Nx.to_number()

      assert abs(mean) < 0.1
      assert std < 0.1
    end
  end

  describe "new/2 with sample_vocab init" do
    test "samples from word embeddings" do
      config = Config.new(
        num_virtual_tokens: 5,
        token_dim: 64,
        prompt_tuning_init: :sample_vocab
      )

      # Mock vocabulary embeddings
      word_embeddings = Nx.random_uniform({1000, 64})
      embedding = Embedding.new(config, word_embeddings)

      assert Nx.shape(embedding.embedding) == {5, 64}
    end
  end

  describe "forward/2" do
    test "returns batch of prompt embeddings" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 4)

      assert Nx.shape(output) == {4, 10, 64}
    end

    test "all batch items are identical" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 4)

      first = output[0]
      second = output[1]

      assert Nx.all_close(first, second) |> Nx.to_number() == 1
    end
  end

  describe "get_trainable_params/1" do
    test "returns embedding tensor" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      params = Embedding.get_trainable_params(embedding)

      assert Map.has_key?(params, "embedding")
      assert Nx.shape(params["embedding"]) == {10, 64}
    end
  end
end
```

### Phase 3: Model Integration

**Step 1: Write failing tests**

```elixir
# test/hf_peft_ex/tuners/prompt_tuning/model_test.exs
defmodule HfPeftEx.Tuners.PromptTuning.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.{Config, Model}

  describe "new/2" do
    test "creates model with prompt encoder" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)

      model = Model.new(base_model, config)

      assert model.config == config
      assert model.prompt_encoder != nil
    end

    test "extracts word embeddings from base model" do
      base_model = create_mock_model()
      config = Config.new(
        num_virtual_tokens: 10,
        token_dim: 64,
        prompt_tuning_init: :sample_vocab
      )

      model = Model.new(base_model, config)

      # Should have sampled from word embeddings
      assert Nx.shape(model.prompt_encoder.embedding) == {10, 64}
    end
  end

  describe "prepare_inputs/3" do
    test "concatenates prompt embeddings with input embeddings" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      input_embeds = Nx.random_normal({2, 10, 64})  # batch=2, seq=10
      {combined, _mask} = Model.prepare_inputs(model, input_embeds)

      # Should be [prompts (5) + input (10)] = 15
      assert Nx.shape(combined) == {2, 15, 64}
    end

    test "updates attention mask for prompts" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      input_embeds = Nx.random_normal({2, 10, 64})
      attention_mask = Nx.broadcast(1, {2, 10})

      {_combined, new_mask} = Model.prepare_inputs(model, input_embeds, attention_mask: attention_mask)

      assert Nx.shape(new_mask) == {2, 15}
      # First 5 positions should be 1 (prompts attend)
      assert Nx.to_number(Nx.sum(new_mask[0][0..4])) == 5
    end

    test "creates attention mask if not provided" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      input_embeds = Nx.random_normal({2, 10, 64})
      {_combined, new_mask} = Model.prepare_inputs(model, input_embeds)

      # Should create all-ones mask for full sequence
      assert Nx.shape(new_mask) == {2, 15}
      assert Nx.to_number(Nx.sum(new_mask)) == 30  # 2 * 15
    end
  end

  describe "get_trainable_params/1" do
    test "returns only prompt embedding parameters" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      model = Model.new(base_model, config)

      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "prompt_encoder.embedding")
      assert map_size(params) == 1
    end
  end

  describe "save_pretrained/2 and from_pretrained/2" do
    @tag :tmp_dir
    test "saves and loads prompt encoder", %{tmp_dir: dir} do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      model = Model.new(base_model, config)

      :ok = Model.save_pretrained(model, dir)

      # Verify files exist
      assert File.exists?(Path.join(dir, "adapter_config.json"))

      # Load and verify
      {:ok, loaded} = Model.from_pretrained(base_model, dir)
      assert Nx.shape(loaded.prompt_encoder.embedding) == {10, 64}
    end
  end

  defp create_mock_model do
    %{
      embeddings: Nx.random_normal({1000, 64}),
      config: %{
        hidden_size: 64,
        vocab_size: 1000
      }
    }
  end
end
```

---

## Quality Requirements

### All Tests Must Pass

```bash
mix test test/hf_peft_ex/tuners/prompt_tuning/
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
## Prompt Tuning

Prompt Tuning adds learnable virtual tokens (soft prompts) to the input. Only the prompt embeddings are trained.

```elixir
config = HfPeftEx.Tuners.PromptTuning.Config.new(
  num_virtual_tokens: 20,
  token_dim: 768,
  prompt_tuning_init: :random
)

# Create prompt tuning model
model = HfPeftEx.Tuners.PromptTuning.Model.new(base_model, config)

# Prepare inputs with prompts
{combined_embeds, attention_mask} = HfPeftEx.Tuners.PromptTuning.Model.prepare_inputs(
  model, input_embeds, attention_mask: mask
)
```

**Parameter count:** `num_virtual_tokens * token_dim` parameters.
```

---

## Mathematical Reference

**Forward:**
```
X_combined = [P_1, ..., P_n, E_1, ..., E_m]
```

Where:
- `P_i` are learnable soft prompt embeddings
- `E_i` are input token embeddings
- Output sequence length is `n + m`

---

## Completion Checklist

- [ ] All config tests pass
- [ ] All embedding tests pass
- [ ] All model tests pass
- [ ] No compiler warnings
- [ ] No dialyzer errors
- [ ] No credo issues
- [ ] Code formatted
- [ ] README.md updated with Prompt Tuning feature
- [ ] Documentation with @doc and @moduledoc
- [ ] Type specs (@spec) on all public functions
