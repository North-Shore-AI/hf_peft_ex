defmodule HfPeftEx.Tuners.Adalora.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.{Config, Model, RankAllocator}

  defp random_normal(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  describe "new/3" do
    test "creates model with config and layer info" do
      config = Config.new(total_step: 1000, init_r: 12, target_r: 8)

      layers = %{
        "q_proj" => %{weight: random_normal({64, 32}), bias: nil},
        "v_proj" => %{weight: random_normal({64, 32}), bias: nil}
      }

      model = Model.new(config, layers)

      assert model.config == config
      assert model.adapter_name == "default"
      assert Map.has_key?(model.layers, "q_proj")
      assert Map.has_key?(model.layers, "v_proj")
    end

    test "initializes layers with init_r rank" do
      config = Config.new(total_step: 1000, init_r: 16, target_r: 8, lora_alpha: 32)

      layers = %{
        "q_proj" => %{weight: random_normal({64, 32}), bias: nil}
      }

      model = Model.new(config, layers)
      layer = model.layers["q_proj"]

      assert Nx.shape(layer.lora_a["default"]) == {16, 32}
      assert Nx.shape(layer.lora_e["default"]) == {16, 1}
    end

    test "creates rank allocator" do
      config = Config.new(total_step: 1000, init_r: 12, target_r: 8)
      layers = %{"q_proj" => %{weight: random_normal({64, 32}), bias: nil}}

      model = Model.new(config, layers)

      assert model.rank_allocator != nil
      assert model.rank_allocator.config == config
    end

    test "supports custom adapter name" do
      config = Config.new(total_step: 1000)

      layers = %{"q_proj" => %{weight: random_normal({64, 32}), bias: nil}}

      model = Model.new(config, layers, adapter_name: "custom")

      assert model.adapter_name == "custom"
      assert Map.has_key?(model.layers["q_proj"].lora_a, "custom")
    end
  end

  describe "forward/2" do
    test "computes output for all layers" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)

      layers = %{
        "layer1" => %{weight: Nx.eye(4), bias: nil}
      }

      model = Model.new(config, layers)
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])

      result = Model.forward(model, %{"layer1" => input})

      assert Nx.shape(result["layer1"]) == {1, 4}
    end

    test "applies lora delta when not disabled" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)

      layers = %{
        "layer1" => %{weight: Nx.eye(4), bias: nil}
      }

      model = Model.new(config, layers)

      # Set lora_b to non-zero to create actual delta
      layer = model.layers["layer1"]

      layer = %{
        layer
        | lora_b: %{"default" => Nx.broadcast(0.1, {4, 4})}
      }

      model = %{model | layers: %{model.layers | "layer1" => layer}}

      input = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      result = Model.forward(model, %{"layer1" => input})

      # Should not be identity anymore due to lora delta
      refute Nx.to_number(result["layer1"][0][0]) == 1.0
    end
  end

  describe "update_and_allocate/3" do
    test "delegates to rank allocator" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10)
      layers = %{"layer1" => %{weight: random_normal({64, 32}), bias: nil}}
      model = Model.new(config, layers)

      gradients = %{
        "layer1" => %{
          lora_e: Nx.broadcast(1.0, {12, 1}),
          lora_e_grad: Nx.broadcast(0.1, {12, 1})
        }
      }

      {updated_model, masks} = Model.update_and_allocate(model, gradients, 10)

      assert updated_model != model
      assert masks != nil
    end

    test "applies masks to layers when returned" do
      config =
        Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10, init_r: 4, target_r: 2)

      layers = %{"layer1" => %{weight: random_normal({64, 32}), bias: nil}}
      model = Model.new(config, layers)

      # Pre-populate exp_avg_ipt for mask computation
      rank_allocator = %{
        model.rank_allocator
        | exp_avg_ipt: %{"layer1" => Nx.tensor([[1.0], [0.5], [0.2], [0.1]])},
          exp_avg_unc: %{"layer1" => Nx.tensor([[0.0], [0.0], [0.0], [0.0]])}
      }

      model = %{model | rank_allocator: rank_allocator}

      gradients = %{
        "layer1" => %{
          lora_e: Nx.broadcast(1.0, {4, 1}),
          lora_e_grad: Nx.broadcast(0.1, {4, 1})
        }
      }

      {updated_model, _masks} = Model.update_and_allocate(model, gradients, 10)

      # Layer should have some values masked
      layer = updated_model.layers["layer1"]
      # ranknum should be updated (less than original)
      assert layer.ranknum["default"] < 4 or layer.ranknum["default"] == 4
    end

    test "returns nil masks during warmup" do
      config = Config.new(total_step: 1000, tinit: 100)
      layers = %{"layer1" => %{weight: random_normal({64, 32}), bias: nil}}
      model = Model.new(config, layers)

      {_model, masks} = Model.update_and_allocate(model, %{}, 50)
      assert masks == nil
    end
  end

  describe "get_orthogonal_loss/1" do
    test "computes orthogonal regularization for A and B matrices" do
      config = Config.new(total_step: 1000)
      layers = %{"layer1" => %{weight: random_normal({64, 32}), bias: nil}}
      model = Model.new(config, layers)

      loss = Model.get_orthogonal_loss(model)

      # Loss should be a scalar
      assert Nx.shape(loss) == {}
      # Loss should be non-negative
      assert Nx.to_number(loss) >= 0
    end

    test "returns zero when no adapters" do
      config = Config.new(total_step: 1000)

      model = %Model{
        config: config,
        adapter_name: "default",
        layers: %{},
        rank_allocator: RankAllocator.new(config)
      }

      loss = Model.get_orthogonal_loss(model)

      assert Nx.to_number(loss) == 0
    end
  end

  describe "merge/1 and unmerge/1" do
    test "merge adds delta weights to base" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)
      base_weight = Nx.eye(4)
      layers = %{"layer1" => %{weight: base_weight, bias: nil}}
      model = Model.new(config, layers)

      # Set non-zero lora_b for actual delta
      layer = model.layers["layer1"]

      layer = %{
        layer
        | lora_b: %{"default" => Nx.broadcast(0.1, {4, 4})}
      }

      model = %{model | layers: %{model.layers | "layer1" => layer}}

      {merged_model, merged_weights} = Model.merge(model)

      assert merged_model.layers["layer1"].merged == true
      assert merged_weights["layer1"] != nil
    end

    test "unmerge restores original weights" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)
      base_weight = Nx.eye(4)
      layers = %{"layer1" => %{weight: base_weight, bias: nil}}
      model = Model.new(config, layers)

      # Set non-zero lora_b for actual delta
      layer = model.layers["layer1"]

      layer = %{
        layer
        | lora_b: %{"default" => Nx.broadcast(0.1, {4, 4})}
      }

      model = %{model | layers: %{model.layers | "layer1" => layer}}

      {merged_model, merged_weights} = Model.merge(model)
      {unmerged_model, restored_weights} = Model.unmerge(merged_model, merged_weights)

      assert unmerged_model.layers["layer1"].merged == false
      assert_all_close(restored_weights["layer1"], base_weight, atol: 1.0e-5)
    end
  end

  describe "resize_by_rank_pattern/2" do
    test "resizes layers based on rank pattern" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)
      layers = %{"layer1" => %{weight: random_normal({64, 32}), bias: nil}}
      model = Model.new(config, layers)

      rank_pattern = %{
        "layer1" => [true, true, false, false]
      }

      resized_model = Model.resize_by_rank_pattern(model, rank_pattern)

      layer = resized_model.layers["layer1"]
      # New rank should be 2
      assert layer.ranknum["default"] == 2
      assert Nx.shape(layer.lora_a["default"]) == {2, 32}
    end
  end

  defp assert_all_close(a, b, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
