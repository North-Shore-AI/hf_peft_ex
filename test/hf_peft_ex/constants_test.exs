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

  describe "tokenizer_config_name/0" do
    test "returns tokenizer_config.json" do
      assert Constants.tokenizer_config_name() == "tokenizer_config.json"
    end
  end

  describe "embedding_layer_names/0" do
    test "returns list of common embedding names" do
      names = Constants.embedding_layer_names()

      assert "embed_tokens" in names
      assert "lm_head" in names
      assert is_list(names)
    end
  end

  describe "seq_cls_head_names/0" do
    test "returns list of classification head names" do
      names = Constants.seq_cls_head_names()

      assert "score" in names
      assert "classifier" in names
      assert is_list(names)
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

    test "returns target modules for bert" do
      modules = Constants.get_lora_target_modules("bert")
      assert modules == ["query", "value"]
    end

    test "returns target modules for t5" do
      modules = Constants.get_lora_target_modules("t5")
      assert modules == ["q", "v"]
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

    test "returns target modules for gpt2" do
      modules = Constants.get_ia3_target_modules("gpt2")
      assert "c_attn" in modules
      assert "mlp.c_proj" in modules
    end

    test "returns nil for unknown model" do
      assert Constants.get_ia3_target_modules("unknown_model") == nil
    end
  end

  describe "get_ia3_feedforward_modules/1" do
    test "returns feedforward modules for llama" do
      modules = Constants.get_ia3_feedforward_modules("llama")
      assert modules == ["down_proj"]
    end

    test "returns feedforward modules for gpt2" do
      modules = Constants.get_ia3_feedforward_modules("gpt2")
      assert modules == ["mlp.c_proj"]
    end

    test "returns nil for unknown model" do
      assert Constants.get_ia3_feedforward_modules("unknown_model") == nil
    end
  end

  describe "get_adalora_target_modules/1" do
    test "returns target modules for llama" do
      modules = Constants.get_adalora_target_modules("llama")
      assert modules == ["q_proj", "v_proj"]
    end

    test "returns more complete modules for t5" do
      modules = Constants.get_adalora_target_modules("t5")
      assert "q" in modules
      assert "k" in modules
      assert "v" in modules
      assert "o" in modules
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
      assert "v_proj" in modules
      assert "down_proj" in modules
    end

    test "returns adalora modules for :adalora type" do
      modules = Constants.get_target_modules(:adalora, "llama")
      assert modules == ["q_proj", "v_proj"]
    end

    test "returns nil for unknown model" do
      assert Constants.get_target_modules(:lora, "unknown_model") == nil
    end

    test "returns nil for unsupported peft type" do
      assert Constants.get_target_modules(:unsupported, "llama") == nil
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

    test "returns list of supported models for :ia3" do
      types = Constants.supported_model_types(:ia3)

      assert "llama" in types
      assert "gpt2" in types
      assert is_list(types)
    end

    test "returns empty list for unsupported type" do
      assert Constants.supported_model_types(:unsupported) == []
    end
  end

  describe "min_target_modules_for_optimization/0" do
    test "returns the optimization threshold" do
      assert Constants.min_target_modules_for_optimization() == 20
    end
  end
end
