# TDD Implementation Prompt: Core Utilities (Mapping, Constants, Save/Load)

## Task Overview

Implement the core utility infrastructure for the `hf_peft_ex` Elixir library using Test-Driven Development.

This prompt covers:
1. **Mapping** - PEFT type to config/tuner class registry
2. **Constants** - Model-to-target-module mappings
3. **Save/Load** - State dict operations for adapter persistence

---

## Required Reading

### Python Source Files (MUST READ FIRST)

1. **Mapping:** `peft/src/peft/mapping.py`
2. **Constants:** `peft/src/peft/utils/constants.py`
3. **Save/Load:** `peft/src/peft/utils/save_and_load.py`
4. **PEFT Types:** `peft/src/peft/utils/peft_types.py`

### Elixir Context Files (READ FOR PATTERNS)

1. **PeftType Enum:** `lib/hf_peft_ex/peft_type.ex`
2. **Config:** `lib/hf_peft_ex/config.ex`
3. **LoRA Config:** `lib/hf_peft_ex/tuners/lora/config.ex`

### Documentation Files

1. **Mapping Doc:** `docs/20251231/peft_gap_analysis/core/01_mapping.md`
2. **Constants Doc:** `docs/20251231/peft_gap_analysis/core/04_constants.md`
3. **Save/Load Doc:** `docs/20251231/peft_gap_analysis/core/03_save_and_load.md`

---

## Part 1: Mapping Module

### Files to Create

1. `lib/hf_peft_ex/mapping.ex`
2. `test/hf_peft_ex/mapping_test.exs`

### TDD Tests

```elixir
# test/hf_peft_ex/mapping_test.exs
defmodule HfPeftEx.MappingTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Mapping

  describe "get_config_class/1" do
    test "returns LoRA config for :lora" do
      assert Mapping.get_config_class(:lora) == HfPeftEx.Tuners.Lora.Config
    end

    test "returns nil for unknown type" do
      assert Mapping.get_config_class(:unknown) == nil
    end

    test "returns config class for all known types" do
      for type <- [:lora, :ia3, :prompt_tuning, :prefix_tuning, :adalora] do
        assert Mapping.get_config_class(type) != nil
      end
    end
  end

  describe "get_tuner_class/1" do
    test "returns LoRA model for :lora" do
      assert Mapping.get_tuner_class(:lora) == HfPeftEx.Tuners.Lora.Model
    end

    test "returns nil for unknown type" do
      assert Mapping.get_tuner_class(:unknown) == nil
    end
  end

  describe "get_prefix/1" do
    test "returns lora_ for :lora" do
      assert Mapping.get_prefix(:lora) == "lora_"
    end

    test "returns ia3_ for :ia3" do
      assert Mapping.get_prefix(:ia3) == "ia3_"
    end
  end

  describe "get_peft_config/1" do
    test "creates config from dict with peft_type" do
      dict = %{
        "peft_type" => "lora",
        "r" => 8,
        "lora_alpha" => 16
      }

      {:ok, config} = Mapping.get_peft_config(dict)

      assert config.__struct__ == HfPeftEx.Tuners.Lora.Config
      assert config.r == 8
      assert config.lora_alpha == 16
    end

    test "returns error for missing peft_type" do
      assert {:error, _} = Mapping.get_peft_config(%{"r" => 8})
    end

    test "returns error for unknown peft_type" do
      assert {:error, _} = Mapping.get_peft_config(%{"peft_type" => "unknown"})
    end
  end

  describe "supported_peft_types/0" do
    test "returns list of all supported types" do
      types = Mapping.supported_peft_types()

      assert :lora in types
      assert is_list(types)
      assert length(types) > 0
    end
  end
end
```

---

## Part 2: Constants Module

### Files to Create

1. `lib/hf_peft_ex/constants.ex`
2. `test/hf_peft_ex/constants_test.exs`

### TDD Tests

```elixir
# test/hf_peft_ex/constants_test.exs
defmodule HfPeftEx.ConstantsTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Constants

  describe "config_name/0" do
    test "returns adapter_config.json" do
      assert Constants.config_name() == "adapter_config.json"
    end
  end

  describe "weights_name/0" do
    test "returns adapter_model.bin" do
      assert Constants.weights_name() == "adapter_model.bin"
    end
  end

  describe "safetensors_weights_name/0" do
    test "returns adapter_model.safetensors" do
      assert Constants.safetensors_weights_name() == "adapter_model.safetensors"
    end
  end

  describe "all_linear_shorthand/0" do
    test "returns all-linear" do
      assert Constants.all_linear_shorthand() == "all-linear"
    end
  end

  describe "embedding_layer_names/0" do
    test "returns list of common embedding names" do
      names = Constants.embedding_layer_names()

      assert "embed_tokens" in names
      assert "lm_head" in names
    end
  end

  describe "get_lora_target_modules/1" do
    test "returns target modules for llama" do
      modules = Constants.get_lora_target_modules("llama")
      assert modules == ["q_proj", "v_proj"]
    end

    test "returns target modules for gpt2" do
      modules = Constants.get_lora_target_modules("gpt2")
      assert modules == ["c_attn"]
    end

    test "returns nil for unknown model" do
      assert Constants.get_lora_target_modules("unknown_model") == nil
    end
  end

  describe "get_ia3_target_modules/1" do
    test "returns target modules for llama" do
      modules = Constants.get_ia3_target_modules("llama")
      assert "k_proj" in modules
      assert "v_proj" in modules
      assert "down_proj" in modules
    end
  end

  describe "get_ia3_feedforward_modules/1" do
    test "returns feedforward modules for llama" do
      modules = Constants.get_ia3_feedforward_modules("llama")
      assert modules == ["down_proj"]
    end
  end

  describe "get_target_modules/2" do
    test "returns lora modules for :lora type" do
      modules = Constants.get_target_modules(:lora, "llama")
      assert modules == ["q_proj", "v_proj"]
    end

    test "returns ia3 modules for :ia3 type" do
      modules = Constants.get_target_modules(:ia3, "llama")
      assert "k_proj" in modules
    end
  end

  describe "supported_model_types/1" do
    test "returns list of supported models for :lora" do
      types = Constants.supported_model_types(:lora)

      assert "llama" in types
      assert "gpt2" in types
      assert "bert" in types
      assert is_list(types)
    end
  end
end
```

---

## Part 3: Save and Load Module

### Files to Create

1. `lib/hf_peft_ex/utils/save_and_load.ex`
2. `test/hf_peft_ex/utils/save_and_load_test.exs`

### TDD Tests

```elixir
# test/hf_peft_ex/utils/save_and_load_test.exs
defmodule HfPeftEx.Utils.SaveAndLoadTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Utils.SaveAndLoad

  describe "get_peft_model_state_dict/3" do
    test "extracts adapter weights only" do
      model = create_mock_lora_model()

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # Should only have lora keys
      for key <- Map.keys(state_dict) do
        assert String.contains?(key, "lora_")
      end
    end

    test "removes adapter name prefix" do
      model = create_mock_lora_model()

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model, "default")

      # Keys should not include adapter name
      for key <- Map.keys(state_dict) do
        refute String.contains?(key, ".default.")
      end
    end

    test "handles bias modes" do
      model = create_mock_lora_model(bias: :lora_only)

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # Should include lora biases but not base biases
      assert Map.has_key?(state_dict, "layer1.lora_B.bias")
    end
  end

  describe "set_peft_model_state_dict/3" do
    test "loads weights into model" do
      model = create_mock_lora_model()
      state_dict = %{
        "layer1.lora_A.weight" => Nx.random_normal({4, 32}),
        "layer1.lora_B.weight" => Nx.random_normal({64, 4})
      }

      {:ok, updated} = SaveAndLoad.set_peft_model_state_dict(model, state_dict)

      # Verify weights were loaded
      assert updated.lora_layers["layer1"].lora_a["default"] != nil
    end

    test "adds adapter name prefix to keys" do
      model = create_mock_lora_model()
      state_dict = %{
        "layer1.lora_A.weight" => Nx.random_normal({4, 32})
      }

      {:ok, updated} = SaveAndLoad.set_peft_model_state_dict(model, state_dict, "custom")

      assert updated.lora_layers["layer1"].lora_a["custom"] != nil
    end

    test "handles mismatched sizes gracefully" do
      model = create_mock_lora_model()
      state_dict = %{
        "layer1.lora_A.weight" => Nx.random_normal({8, 32})  # Wrong rank
      }

      result = SaveAndLoad.set_peft_model_state_dict(model, state_dict, "default",
        ignore_mismatched_sizes: true)

      assert {:ok, _} = result
    end
  end

  describe "save_peft_weights/3" do
    @tag :tmp_dir
    test "saves weights to safetensors file", %{tmp_dir: dir} do
      state_dict = %{
        "layer1.lora_A.weight" => Nx.random_normal({4, 32}),
        "layer1.lora_B.weight" => Nx.random_normal({64, 4})
      }

      path = Path.join(dir, "adapter_model.safetensors")
      :ok = SaveAndLoad.save_peft_weights(state_dict, path)

      assert File.exists?(path)
    end

    @tag :tmp_dir
    test "saves to Nx format when safetensors unavailable", %{tmp_dir: dir} do
      state_dict = %{
        "layer1.lora_A.weight" => Nx.random_normal({4, 32})
      }

      path = Path.join(dir, "adapter_model.nx")
      :ok = SaveAndLoad.save_peft_weights(state_dict, path, format: :nx)

      assert File.exists?(path)
    end
  end

  describe "load_peft_weights/2" do
    @tag :tmp_dir
    test "loads weights from file", %{tmp_dir: dir} do
      original = %{
        "layer1.lora_A.weight" => Nx.random_normal({4, 32})
      }

      path = Path.join(dir, "adapter_model.nx")
      :ok = SaveAndLoad.save_peft_weights(original, path, format: :nx)

      {:ok, loaded} = SaveAndLoad.load_peft_weights(path)

      assert Map.has_key?(loaded, "layer1.lora_A.weight")
      assert Nx.shape(loaded["layer1.lora_A.weight"]) == {4, 32}
    end

    test "returns error for missing file" do
      assert {:error, _} = SaveAndLoad.load_peft_weights("/nonexistent/path")
    end
  end

  describe "filter_adapter_keys/3" do
    test "filters to only adapter-related keys" do
      state_dict = %{
        "base_model.layer1.weight" => Nx.tensor([1.0]),
        "base_model.layer1.lora_A.default.weight" => Nx.tensor([2.0]),
        "base_model.layer1.lora_B.default.weight" => Nx.tensor([3.0])
      }

      filtered = SaveAndLoad.filter_adapter_keys(state_dict, "default", :lora)

      assert map_size(filtered) == 2
      for key <- Map.keys(filtered) do
        assert String.contains?(key, "lora_")
      end
    end
  end

  defp create_mock_lora_model(opts \\ []) do
    %{
      lora_layers: %{
        "layer1" => %{
          lora_a: %{"default" => Nx.random_normal({4, 32})},
          lora_b: %{"default" => Nx.random_normal({64, 4})},
          lora_bias: if(opts[:bias], do: %{"default" => Nx.broadcast(0.0, {64})}, else: nil)
        }
      },
      config: %HfPeftEx.Tuners.Lora.Config{
        peft_type: :lora,
        bias: Keyword.get(opts, :bias, :none)
      },
      active_adapter: "default"
    }
  end
end
```

---

## Quality Requirements

### All Tests Must Pass

```bash
mix test test/hf_peft_ex/mapping_test.exs test/hf_peft_ex/constants_test.exs test/hf_peft_ex/utils/save_and_load_test.exs
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
## Core Utilities

### Mapping Registry

```elixir
# Get config class for PEFT type
HfPeftEx.Mapping.get_config_class(:lora)
# => HfPeftEx.Tuners.Lora.Config

# Create config from dict
{:ok, config} = HfPeftEx.Mapping.get_peft_config(%{
  "peft_type" => "lora",
  "r" => 8
})
```

### Target Module Constants

```elixir
# Get default target modules for model type
HfPeftEx.Constants.get_lora_target_modules("llama")
# => ["q_proj", "v_proj"]

HfPeftEx.Constants.get_target_modules(:ia3, "gpt2")
# => ["c_attn", "mlp.c_proj"]
```

### Save and Load

```elixir
# Save adapter weights
{:ok, state_dict} = HfPeftEx.Utils.SaveAndLoad.get_peft_model_state_dict(model)
:ok = HfPeftEx.Utils.SaveAndLoad.save_peft_weights(state_dict, "adapter.safetensors")

# Load adapter weights
{:ok, weights} = HfPeftEx.Utils.SaveAndLoad.load_peft_weights("adapter.safetensors")
{:ok, model} = HfPeftEx.Utils.SaveAndLoad.set_peft_model_state_dict(model, weights)
```
```

---

## Completion Checklist

- [ ] All mapping tests pass
- [ ] All constants tests pass
- [ ] All save_and_load tests pass
- [ ] No compiler warnings
- [ ] No dialyzer errors
- [ ] No credo issues
- [ ] Code formatted
- [ ] README.md updated with utilities
- [ ] Documentation with @doc and @moduledoc
- [ ] Type specs (@spec) on all public functions
