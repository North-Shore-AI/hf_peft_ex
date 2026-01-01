defmodule HfPeftEx.Tuners.IA3.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.{Config, Model}

  # Helper to generate random tensors
  defp random_normal(shape) do
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  defp create_mock_model do
    %{
      layer1: %{weight: random_normal({64, 32}), bias: nil},
      layer2: %{weight: random_normal({64, 64}), bias: nil},
      layer3: %{weight: random_normal({128, 64}), bias: nil}
    }
  end

  describe "new/2" do
    test "wraps base model with IA3 config" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1", "layer2"])

      model = Model.new(base_model, config)

      assert model.config == config
      assert model.base_model == base_model
      assert model.adapter_name == "default"
    end

    test "accepts adapter_name option" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])

      model = Model.new(base_model, config, adapter_name: "custom_adapter")

      assert model.adapter_name == "custom_adapter"
    end

    test "initializes as not merged" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])

      model = Model.new(base_model, config)

      assert model.merged == false
    end

    test "initializes with adapters enabled" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])

      model = Model.new(base_model, config)

      assert model.adapters_enabled == true
    end
  end

  describe "target_modules_match?/2" do
    test "matches module in list" do
      config = Config.new(target_modules: ["q_proj", "v_proj"])

      assert Model.target_modules_match?(config, "q_proj") == true
      assert Model.target_modules_match?(config, "v_proj") == true
      assert Model.target_modules_match?(config, "k_proj") == false
    end

    test "matches module with regex pattern" do
      config = Config.new(target_modules: ".*_proj")

      assert Model.target_modules_match?(config, "q_proj") == true
      assert Model.target_modules_match?(config, "v_proj") == true
      assert Model.target_modules_match?(config, "attention") == false
    end

    test "returns false for nil target_modules" do
      config = Config.new(target_modules: nil)

      assert Model.target_modules_match?(config, "q_proj") == false
    end
  end

  describe "feedforward_modules_match?/2" do
    test "matches feedforward module in list" do
      config =
        Config.new(
          target_modules: ["q_proj", "down_proj"],
          feedforward_modules: ["down_proj"]
        )

      assert Model.feedforward_modules_match?(config, "down_proj") == true
      assert Model.feedforward_modules_match?(config, "q_proj") == false
    end

    test "returns false for nil feedforward_modules" do
      config = Config.new(target_modules: ["q_proj"])

      assert Model.feedforward_modules_match?(config, "q_proj") == false
    end
  end

  describe "exclude_modules_match?/2" do
    test "matches excluded module in list" do
      config = Config.new(target_modules: ["q_proj"], exclude_modules: ["lm_head"])

      assert Model.exclude_modules_match?(config, "lm_head") == true
      assert Model.exclude_modules_match?(config, "q_proj") == false
    end

    test "returns false for nil exclude_modules" do
      config = Config.new(target_modules: ["q_proj"])

      assert Model.exclude_modules_match?(config, "lm_head") == false
    end
  end

  describe "should_apply_ia3?/2" do
    test "returns true for matching target not excluded" do
      config = Config.new(target_modules: ["q_proj", "v_proj"])

      assert Model.should_apply_ia3?(config, "q_proj") == true
    end

    test "returns false for excluded module" do
      config = Config.new(target_modules: ["q_proj", "lm_head"], exclude_modules: ["lm_head"])

      assert Model.should_apply_ia3?(config, "lm_head") == false
    end

    test "returns false for non-matching module" do
      config = Config.new(target_modules: ["q_proj"])

      assert Model.should_apply_ia3?(config, "k_proj") == false
    end
  end

  describe "add_ia3_layers/2" do
    test "adds IA3 layers for matching modules" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1", "layer2"])
      model = Model.new(base_model, config)

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 32, out_features: 64},
        "layer2" => %{type: :linear, in_features: 64, out_features: 64},
        "layer3" => %{type: :linear, in_features: 64, out_features: 128}
      }

      model = Model.add_ia3_layers(model, module_specs)

      assert Map.has_key?(model.ia3_layers, "layer1")
      assert Map.has_key?(model.ia3_layers, "layer2")
      refute Map.has_key?(model.ia3_layers, "layer3")
    end

    test "identifies feedforward modules correctly" do
      base_model = create_mock_model()

      config =
        Config.new(
          target_modules: ["layer1", "layer2"],
          feedforward_modules: ["layer2"]
        )

      model = Model.new(base_model, config)

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 32, out_features: 64},
        "layer2" => %{type: :linear, in_features: 64, out_features: 64}
      }

      model = Model.add_ia3_layers(model, module_specs)

      refute model.ia3_layers["layer1"].is_feedforward
      assert model.ia3_layers["layer2"].is_feedforward
    end
  end

  describe "set_adapter/2" do
    test "switches active adapter" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)

      model = Model.set_adapter(model, "adapter2")

      assert model.adapter_name == "adapter2"
    end
  end

  describe "enable_adapter/2 and disable_adapter/2" do
    test "enables adapters" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)
      model = %{model | adapters_enabled: false}

      model = Model.enable_adapter(model, "default")

      assert model.adapters_enabled == true
    end

    test "disables adapters" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)

      model = Model.disable_adapter(model, "default")

      assert model.adapters_enabled == false
    end
  end

  describe "get_trainable_params/1" do
    test "returns only ia3_l parameters" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1", "layer2"])
      model = Model.new(base_model, config)

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 32, out_features: 64},
        "layer2" => %{type: :linear, in_features: 64, out_features: 64}
      }

      model = Model.add_ia3_layers(model, module_specs)
      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "layer1.ia3_l")
      assert Map.has_key?(params, "layer2.ia3_l")
      assert map_size(params) == 2

      # Check shapes
      assert Nx.shape(params["layer1.ia3_l"]) == {64}
      assert Nx.shape(params["layer2.ia3_l"]) == {64}
    end
  end

  describe "merge_adapter/1 and unmerge_adapter/1" do
    test "merge sets merged flag" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)

      model = Model.merge_adapter(model)

      assert model.merged == true
    end

    test "unmerge clears merged flag" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)
      model = %{model | merged: true}

      model = Model.unmerge_adapter(model)

      assert model.merged == false
    end
  end

  describe "get_peft_config/1" do
    test "returns the config" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)

      assert Model.get_peft_config(model) == config
    end
  end

  describe "apply_adapters/3" do
    test "applies IA3 scaling to module outputs" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 4, out_features: 4}
      }

      model = Model.add_ia3_layers(model, module_specs)

      # Set scaling to 2.0
      layer = model.ia3_layers["layer1"]
      layer = %{layer | ia3_l: Nx.tensor([2.0, 2.0, 2.0, 2.0])}
      model = put_in(model.ia3_layers["layer1"], layer)

      module_inputs = %{
        "layer1" => %{
          input: Nx.tensor([[1.0, 2.0, 3.0, 4.0]]),
          base_output: Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
        }
      }

      outputs = Model.apply_adapters(model, module_inputs)

      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0]])
      assert_all_close(outputs["layer1"], expected)
    end

    test "returns base output when adapters disabled" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)
      model = %{model | adapters_enabled: false}

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 4, out_features: 4}
      }

      model = Model.add_ia3_layers(model, module_specs)

      module_inputs = %{
        "layer1" => %{
          input: Nx.tensor([[1.0, 2.0, 3.0, 4.0]]),
          base_output: Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
        }
      }

      outputs = Model.apply_adapters(model, module_inputs)

      assert_all_close(outputs["layer1"], Nx.tensor([[1.0, 2.0, 3.0, 4.0]]))
    end

    test "returns base output when merged" do
      base_model = create_mock_model()
      config = Config.new(target_modules: ["layer1"])
      model = Model.new(base_model, config)
      model = %{model | merged: true}

      module_specs = %{
        "layer1" => %{type: :linear, in_features: 4, out_features: 4}
      }

      model = Model.add_ia3_layers(model, module_specs)

      module_inputs = %{
        "layer1" => %{
          input: Nx.tensor([[1.0, 2.0, 3.0, 4.0]]),
          base_output: Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
        }
      }

      outputs = Model.apply_adapters(model, module_inputs)

      assert_all_close(outputs["layer1"], Nx.tensor([[1.0, 2.0, 3.0, 4.0]]))
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
