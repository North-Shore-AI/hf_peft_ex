defmodule HfPeftEx.Tuners.Lora.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Conv1d
  alias HfPeftEx.Tuners.Lora.Conv2d
  alias HfPeftEx.Tuners.Lora.Embedding
  alias HfPeftEx.Tuners.Lora.Linear
  alias HfPeftEx.Tuners.Lora.Model

  describe "new/2" do
    test "creates a LoRA model wrapper" do
      config = LoraConfig.new(r: 8, target_modules: ["dense"])
      model = Model.new(%{type: :test_model, layers: []}, config)

      assert model.config == config
      assert model.adapter_name == "default"
    end

    test "stores the base model" do
      config = LoraConfig.new(r: 8)
      base_model = %{type: :transformer, layers: [:layer1, :layer2]}
      model = Model.new(base_model, config)

      assert model.base_model == base_model
    end

    test "accepts custom adapter name" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config, adapter_name: "my_adapter")

      assert model.adapter_name == "my_adapter"
    end

    test "initializes with adapters enabled" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config)

      assert model.adapters_enabled == true
    end
  end

  describe "target_modules_match?/2" do
    test "matches exact module name in list" do
      config = LoraConfig.new(target_modules: ["q_proj", "v_proj"])

      assert Model.target_modules_match?(config, "q_proj") == true
      assert Model.target_modules_match?(config, "v_proj") == true
      assert Model.target_modules_match?(config, "k_proj") == false
    end

    test "matches with regex pattern" do
      config = LoraConfig.new(target_modules: ".*_proj")

      assert Model.target_modules_match?(config, "q_proj") == true
      assert Model.target_modules_match?(config, "v_proj") == true
      assert Model.target_modules_match?(config, "dense") == false
    end

    test "matches all-linear keyword" do
      config = LoraConfig.new(target_modules: "all-linear")

      # All module names should match when using "all-linear"
      assert Model.target_modules_match?(config, "q_proj") == true
      assert Model.target_modules_match?(config, "dense") == true
    end

    test "returns false when target_modules is nil" do
      config = LoraConfig.new(target_modules: nil)

      assert Model.target_modules_match?(config, "anything") == false
    end
  end

  describe "exclude_modules_match?/2" do
    test "matches exact module name in exclude list" do
      config = LoraConfig.new(exclude_modules: ["lm_head", "embed"])

      assert Model.exclude_modules_match?(config, "lm_head") == true
      assert Model.exclude_modules_match?(config, "embed") == true
      assert Model.exclude_modules_match?(config, "q_proj") == false
    end

    test "returns false when exclude_modules is nil" do
      config = LoraConfig.new(exclude_modules: nil)

      assert Model.exclude_modules_match?(config, "anything") == false
    end
  end

  describe "should_apply_lora?/2" do
    test "returns true when target matches and not excluded" do
      config = LoraConfig.new(target_modules: ["q_proj", "v_proj"])

      assert Model.should_apply_lora?(config, "q_proj") == true
    end

    test "returns false when excluded even if target matches" do
      config =
        LoraConfig.new(
          target_modules: ["q_proj", "v_proj", "lm_head"],
          exclude_modules: ["lm_head"]
        )

      assert Model.should_apply_lora?(config, "q_proj") == true
      assert Model.should_apply_lora?(config, "lm_head") == false
    end
  end

  describe "enable_adapter/2 and disable_adapter/2" do
    test "enable_adapter sets adapters_enabled to true" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config)
      model = %{model | adapters_enabled: false}

      model = Model.enable_adapter(model, "default")
      assert model.adapters_enabled == true
    end

    test "disable_adapter sets adapters_enabled to false" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config)

      model = Model.disable_adapter(model, "default")
      assert model.adapters_enabled == false
    end
  end

  describe "get_peft_config/1" do
    test "returns the LoRA config" do
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      model = Model.new(%{}, config)

      retrieved = Model.get_peft_config(model)
      assert retrieved.r == 16
      assert retrieved.lora_alpha == 32
    end
  end

  describe "merge_adapter/1 and unmerge_adapter/1" do
    test "merge_adapter sets merged flag" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config)

      model = Model.merge_adapter(model)
      assert model.merged == true
    end

    test "unmerge_adapter clears merged flag" do
      config = LoraConfig.new(r: 8)
      model = Model.new(%{}, config)
      model = %{model | merged: true}

      model = Model.unmerge_adapter(model)
      assert model.merged == false
    end
  end

  describe "add_lora_layers/2" do
    test "builds lora layers for targeted modules" do
      config =
        LoraConfig.new(
          r: 4,
          lora_alpha: 8,
          target_modules: ["q_proj", "embed", "conv1", "conv2"],
          use_dora: true
        )

      model = Model.new(%{}, config)

      specs = %{
        "q_proj" => %{type: :linear, in_features: 8, out_features: 4},
        "embed" => %{type: :embedding, num_embeddings: 10, embedding_dim: 4},
        "conv1" => %{type: :conv1d, in_channels: 2, out_channels: 3, kernel_size: 2},
        "conv2" => %{type: :conv2d, in_channels: 1, out_channels: 2, kernel_size: {2, 2}},
        "mlp" => %{type: :linear, in_features: 4, out_features: 4}
      }

      model = Model.add_lora_layers(model, specs)

      assert %Linear{} = model.lora_layers["q_proj"]
      assert %Embedding{} = model.lora_layers["embed"]
      assert %Conv1d{} = model.lora_layers["conv1"]
      assert %Conv2d{} = model.lora_layers["conv2"]
      refute Map.has_key?(model.lora_layers, "mlp")
      assert model.lora_layers["q_proj"].use_dora == true
      assert model.lora_layers["embed"].use_dora == true
    end

    test "respects exclude_modules" do
      config =
        LoraConfig.new(
          target_modules: ".*_proj",
          exclude_modules: ["k_proj"]
        )

      model = Model.new(%{}, config)

      specs = %{
        "q_proj" => %{type: :linear, in_features: 8, out_features: 4},
        "k_proj" => %{type: :linear, in_features: 8, out_features: 4}
      }

      model = Model.add_lora_layers(model, specs)

      assert Map.has_key?(model.lora_layers, "q_proj")
      refute Map.has_key?(model.lora_layers, "k_proj")
    end

    test "propagates fan_in_fan_out for linear layers" do
      config =
        LoraConfig.new(
          target_modules: ["q_proj"],
          fan_in_fan_out: true
        )

      model = Model.new(%{}, config)

      specs = %{
        "q_proj" => %{type: :linear, in_features: 8, out_features: 4}
      }

      model = Model.add_lora_layers(model, specs)

      assert model.lora_layers["q_proj"].fan_in_fan_out == true
    end

    test "skips non-linear layers when target_modules is all-linear" do
      config = LoraConfig.new(target_modules: "all-linear")
      model = Model.new(%{}, config)

      specs = %{
        "linear" => %{type: :linear, in_features: 4, out_features: 4},
        "conv1" => %{type: :conv1d, in_channels: 2, out_channels: 3, kernel_size: 2},
        "embed" => %{type: :embedding, num_embeddings: 10, embedding_dim: 4}
      }

      model = Model.add_lora_layers(model, specs)

      assert Map.has_key?(model.lora_layers, "linear")
      refute Map.has_key?(model.lora_layers, "conv1")
      refute Map.has_key?(model.lora_layers, "embed")
    end
  end

  describe "apply_adapters/3" do
    test "applies linear and embedding adapters" do
      config = LoraConfig.new(r: 2, lora_alpha: 2, target_modules: ["linear", "embed"])
      model = Model.new(%{}, config)

      specs = %{
        "linear" => %{type: :linear, in_features: 2, out_features: 2},
        "embed" => %{type: :embedding, num_embeddings: 3, embedding_dim: 2}
      }

      model = Model.add_lora_layers(model, specs)

      lora_a = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      linear = %{model.lora_layers["linear"] | lora_A: lora_a, lora_B: lora_b}

      lora_emb_a = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      lora_emb_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])

      embed = %{
        model.lora_layers["embed"]
        | lora_embedding_A: lora_emb_a,
          lora_embedding_B: lora_emb_b
      }

      model = %{model | lora_layers: %{"linear" => linear, "embed" => embed}}

      x = Nx.tensor([[1.0, 1.0]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(base_weight))

      indices = Nx.tensor([[0, 1]])
      embed_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      embed_output = Embedding.embed_lookup(indices, embed_weight)

      inputs = %{
        "linear" => %{input: x, base_output: base_output, weight: base_weight},
        "embed" => %{input: indices, base_output: embed_output, weight: embed_weight}
      }

      outputs = Model.apply_adapters(model, inputs)

      expected_linear = Linear.forward(linear, x, base_output, weight: base_weight)
      expected_embed = Embedding.forward(embed, indices, embed_output, weight: embed_weight)

      assert_tensor_close(outputs["linear"], expected_linear)
      assert_tensor_close(outputs["embed"], expected_embed)
    end

    test "applies conv adapters" do
      config =
        LoraConfig.new(r: 2, lora_alpha: 2, target_modules: ["conv1", "conv2"], use_dora: true)

      model = Model.new(%{}, config)

      specs = %{
        "conv1" => %{type: :conv1d, in_channels: 1, out_channels: 1, kernel_size: 2},
        "conv2" => %{type: :conv2d, in_channels: 1, out_channels: 1, kernel_size: {2, 2}}
      }

      model = Model.add_lora_layers(model, specs)

      lora_a1 = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_b1 = Nx.tensor([[0.5, 0.0]])
      conv1 = %{model.lora_layers["conv1"] | lora_A: lora_a1, lora_B: lora_b1}

      lora_a2 =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b2 = Nx.tensor([[0.5, 0.0]])
      conv2 = %{model.lora_layers["conv2"] | lora_A: lora_a2, lora_B: lora_b2}

      model = %{model | lora_layers: %{"conv1" => conv1, "conv2" => conv2}}

      x1 = Nx.tensor([[[1.0, 2.0, 3.0]]])
      w1 = Nx.tensor([[[1.0, 0.0]]])
      base_out1 = Nx.conv(x1, w1)

      x2 = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])
      w2 = Nx.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
      base_out2 = Nx.conv(x2, w2)

      inputs = %{
        "conv1" => %{input: x1, base_output: base_out1, weight: w1},
        "conv2" => %{input: x2, base_output: base_out2, weight: w2}
      }

      outputs = Model.apply_adapters(model, inputs)

      expected_conv1 = Conv1d.forward(conv1, x1, base_out1, weight: w1)
      expected_conv2 = Conv2d.forward(conv2, x2, base_out2, weight: w2)

      assert_tensor_close(outputs["conv1"], expected_conv1)
      assert_tensor_close(outputs["conv2"], expected_conv2)
    end

    test "mixes adapters with partial targets and exclusions" do
      config =
        LoraConfig.new(
          r: 2,
          lora_alpha: 2,
          target_modules: ["linear", "conv1", "embed"],
          exclude_modules: ["embed"],
          use_dora: true
        )

      model = Model.new(%{}, config)

      specs = %{
        "linear" => %{type: :linear, in_features: 2, out_features: 2},
        "embed" => %{type: :embedding, num_embeddings: 3, embedding_dim: 2},
        "conv1" => %{type: :conv1d, in_channels: 1, out_channels: 1, kernel_size: 2},
        "conv2" => %{type: :conv2d, in_channels: 1, out_channels: 1, kernel_size: {2, 2}}
      }

      model = Model.add_lora_layers(model, specs)
      refute Map.has_key?(model.lora_layers, "embed")
      refute Map.has_key?(model.lora_layers, "conv2")

      lora_a = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      linear = %{model.lora_layers["linear"] | lora_A: lora_a, lora_B: lora_b}

      lora_a1 = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_b1 = Nx.tensor([[0.5, 0.0]])
      conv1 = %{model.lora_layers["conv1"] | lora_A: lora_a1, lora_B: lora_b1}

      model = %{model | lora_layers: %{"linear" => linear, "conv1" => conv1}}

      x = Nx.tensor([[1.0, 1.0]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(base_weight))

      indices = Nx.tensor([[0, 1]])
      embed_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      embed_output = Embedding.embed_lookup(indices, embed_weight)

      x1 = Nx.tensor([[[1.0, 2.0, 3.0]]])
      w1 = Nx.tensor([[[1.0, 0.0]]])
      base_out1 = Nx.conv(x1, w1)

      x2 = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])
      w2 = Nx.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
      base_out2 = Nx.conv(x2, w2)

      inputs = %{
        "linear" => %{input: x, base_output: base_output, weight: base_weight},
        "embed" => %{input: indices, base_output: embed_output, weight: embed_weight},
        "conv1" => %{input: x1, base_output: base_out1, weight: w1},
        "conv2" => %{input: x2, base_output: base_out2, weight: w2}
      }

      outputs = Model.apply_adapters(model, inputs)

      expected_linear = Linear.forward(linear, x, base_output, weight: base_weight)
      expected_conv1 = Conv1d.forward(conv1, x1, base_out1, weight: w1)

      assert_tensor_close(outputs["linear"], expected_linear)
      assert_tensor_close(outputs["conv1"], expected_conv1)
      assert_tensor_close(outputs["embed"], embed_output)
      assert_tensor_close(outputs["conv2"], base_out2)
    end

    test "returns base outputs when adapters are disabled or merged for convs" do
      config = LoraConfig.new(r: 2, lora_alpha: 2, target_modules: ["conv1", "conv2"])

      model_disabled = Model.new(%{}, config)
      model_merged = Model.new(%{}, config)

      specs = %{
        "conv1" => %{type: :conv1d, in_channels: 1, out_channels: 1, kernel_size: 2},
        "conv2" => %{type: :conv2d, in_channels: 1, out_channels: 1, kernel_size: {2, 2}}
      }

      model_disabled = Model.add_lora_layers(model_disabled, specs)
      model_disabled = %{model_disabled | adapters_enabled: false}

      model_merged = Model.add_lora_layers(model_merged, specs)
      model_merged = %{model_merged | merged: true}

      x1 = Nx.tensor([[[1.0, 2.0, 3.0]]])
      w1 = Nx.tensor([[[1.0, 0.0]]])
      base_out1 = Nx.conv(x1, w1)

      x2 = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])
      w2 = Nx.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
      base_out2 = Nx.conv(x2, w2)

      inputs = %{
        "conv1" => %{input: x1, base_output: base_out1, weight: w1},
        "conv2" => %{input: x2, base_output: base_out2, weight: w2}
      }

      outputs_disabled = Model.apply_adapters(model_disabled, inputs)
      outputs_merged = Model.apply_adapters(model_merged, inputs)

      assert_tensor_close(outputs_disabled["conv1"], base_out1)
      assert_tensor_close(outputs_disabled["conv2"], base_out2)
      assert_tensor_close(outputs_merged["conv1"], base_out1)
      assert_tensor_close(outputs_merged["conv2"], base_out2)
    end

    test "returns base outputs when adapters are disabled" do
      config = LoraConfig.new(r: 2, target_modules: ["linear"])
      model = Model.new(%{}, config)

      specs = %{"linear" => %{type: :linear, in_features: 2, out_features: 2}}
      model = Model.add_lora_layers(model, specs)
      model = %{model | adapters_enabled: false}

      x = Nx.tensor([[1.0, 1.0]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(base_weight))

      inputs = %{"linear" => %{input: x, base_output: base_output, weight: base_weight}}
      outputs = Model.apply_adapters(model, inputs)

      assert_tensor_close(outputs["linear"], base_output)
    end

    test "returns base outputs when model is merged" do
      config = LoraConfig.new(r: 2, target_modules: ["linear"])
      model = Model.new(%{}, config)

      specs = %{"linear" => %{type: :linear, in_features: 2, out_features: 2}}
      model = Model.add_lora_layers(model, specs)
      model = %{model | merged: true}

      x = Nx.tensor([[1.0, 1.0]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(base_weight))

      inputs = %{"linear" => %{input: x, base_output: base_output, weight: base_weight}}
      outputs = Model.apply_adapters(model, inputs)

      assert_tensor_close(outputs["linear"], base_output)
    end
  end

  defp assert_tensor_close(left, right, tol \\ 1.0e-6) do
    diff = Nx.subtract(left, right) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
    assert diff < tol
  end
end
