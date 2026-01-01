defmodule HfPeftEx.Tuners.IA3.LayerTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.Layer

  # Helper to generate random tensors
  defp random_normal(shape) do
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  describe "new/3" do
    test "creates layer with correct dimensions from base layer" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.in_features == 32
      assert layer.out_features == 64
      assert layer.is_feedforward == false
    end

    test "handles fan_in_fan_out true" do
      # When fan_in_fan_out is true, weight is stored as {in_features, out_features}
      base_layer = %{weight: random_normal({32, 64})}
      layer = Layer.new(base_layer, false, fan_in_fan_out: true)

      assert layer.in_features == 32
      assert layer.out_features == 64
    end

    test "marks layer as feedforward" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, true)

      assert layer.is_feedforward == true
    end

    test "stores base layer reference" do
      base_layer = %{weight: random_normal({64, 32}), bias: random_normal({64})}
      layer = Layer.new(base_layer, false)

      assert layer.base_layer == base_layer
    end

    test "initializes with empty ia3_l map" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.ia3_l == %{}
    end

    test "initializes as not merged" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.merged == false
    end

    test "initializes with adapters enabled" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.disable_adapters == false
    end

    test "sets default active adapter to default" do
      base_layer = %{weight: random_normal({64, 32})}
      layer = Layer.new(base_layer, false)

      assert layer.active_adapter == "default"
    end
  end

  describe "update_layer/3" do
    test "initializes ia3_l to ones when init_ia3_weights is true" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      ia3_l = layer.ia3_l["default"]
      assert Nx.shape(ia3_l) == {64}
      # Check all values are 1.0
      assert Nx.to_number(Nx.mean(ia3_l)) == 1.0
      assert Nx.to_number(Nx.reduce_min(ia3_l)) == 1.0
      assert Nx.to_number(Nx.reduce_max(ia3_l)) == 1.0
    end

    test "uses out_features for non-feedforward layer" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      # Non-feedforward: scaling vector has out_features dimension
      assert Nx.shape(layer.ia3_l["default"]) == {64}
    end

    test "uses in_features for feedforward layer" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, true)
        |> Layer.update_layer("default", true)

      # Feedforward: scaling vector has in_features dimension
      assert Nx.shape(layer.ia3_l["default"]) == {32}
    end

    test "initializes ia3_l with small random values when init_ia3_weights is false" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", false)

      ia3_l = layer.ia3_l["default"]
      assert Nx.shape(ia3_l) == {64}
      # Random values - mean should not be exactly 1.0
      mean = Nx.to_number(Nx.mean(ia3_l))
      assert mean != 1.0
    end

    test "can add multiple adapters" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("adapter1", true)
        |> Layer.update_layer("adapter2", true)

      assert Map.has_key?(layer.ia3_l, "adapter1")
      assert Map.has_key?(layer.ia3_l, "adapter2")
    end
  end

  describe "forward/2" do
    test "returns identity when ia3_l is ones" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      # With identity weight and ia3_l=ones, output should equal input
      assert_all_close(output, input)
    end

    test "scales output by ia3_l for non-feedforward" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      # Manually set scaling to 2.0 for all elements
      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0]])
      assert_all_close(output, expected)
    end

    test "applies different scaling per output dimension" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      # Different scaling for each dimension
      layer = put_in(layer.ia3_l["default"], Nx.tensor([1.0, 2.0, 0.5, 3.0]))

      input = Nx.tensor([[2.0, 2.0, 2.0, 2.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[2.0, 4.0, 1.0, 6.0]])
      assert_all_close(output, expected)
    end

    test "returns base output when adapters disabled" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)
        |> Map.put(:disable_adapters, true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      # Should return base output (not scaled)
      assert_all_close(output, input)
    end

    test "returns base output when merged" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)
        |> Map.put(:merged, true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      # When merged, weights already contain scaling
      assert_all_close(output, input)
    end

    test "handles batch input" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
      assert_all_close(output, expected)
    end

    test "scales output including bias" do
      base_layer = %{weight: Nx.eye(4), bias: Nx.tensor([1.0, 1.0, 1.0, 1.0])}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))

      input = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      output = Layer.forward(layer, input)

      # IA3 scales entire output: (input + bias) * 2 = (1 + 1) * 2 = 4
      expected = Nx.tensor([[4.0, 4.0, 4.0, 4.0]])
      assert_all_close(output, expected)
    end

    test "uses active adapter for scaling" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("adapter1", true)
        |> Layer.update_layer("adapter2", true)

      layer = put_in(layer.ia3_l["adapter1"], Nx.tensor([2.0, 2.0, 2.0, 2.0]))
      layer = put_in(layer.ia3_l["adapter2"], Nx.tensor([3.0, 3.0, 3.0, 3.0]))
      layer = %{layer | active_adapter: "adapter2"}

      input = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[3.0, 3.0, 3.0, 3.0]])
      assert_all_close(output, expected)
    end
  end

  describe "merge/2" do
    test "modifies base weights correctly for non-feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)

      # Rows should be scaled: W[i, :] *= ia3_l[i]
      expected =
        Nx.tensor([
          [2.0, 2.0, 2.0, 2.0],
          [1.0, 1.0, 1.0, 1.0],
          [0.5, 0.5, 0.5, 0.5],
          [1.0, 1.0, 1.0, 1.0]
        ])

      assert_all_close(merged.base_layer.weight, expected)
      assert merged.merged == true
    end

    test "modifies base weights correctly for feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, true)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)

      # Columns should be scaled: W[:, i] *= ia3_l[i]
      expected =
        Nx.tensor([
          [2.0, 1.0, 0.5, 1.0],
          [2.0, 1.0, 0.5, 1.0],
          [2.0, 1.0, 0.5, 1.0],
          [2.0, 1.0, 0.5, 1.0]
        ])

      assert_all_close(merged.base_layer.weight, expected)
      assert merged.merged == true
    end

    test "adds adapter to merged_adapters list" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      {:ok, merged} = Layer.merge(layer)

      assert "default" in merged.merged_adapters
    end

    test "returns error when already merged" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      {:ok, merged} = Layer.merge(layer)
      result = Layer.merge(merged)

      assert {:error, :already_merged} = result
    end

    test "also scales bias for non-feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: Nx.tensor([1.0, 1.0, 1.0, 1.0])}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)

      expected_bias = Nx.tensor([2.0, 1.0, 0.5, 1.0])
      assert_all_close(merged.base_layer.bias, expected_bias)
    end
  end

  describe "unmerge/1" do
    test "restores original weights for non-feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)
      {:ok, unmerged} = Layer.unmerge(merged)

      assert_all_close(unmerged.base_layer.weight, Nx.broadcast(1.0, {4, 4}))
      assert unmerged.merged == false
    end

    test "restores original weights for feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, true)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)
      {:ok, unmerged} = Layer.unmerge(merged)

      assert_all_close(unmerged.base_layer.weight, Nx.broadcast(1.0, {4, 4}))
      assert unmerged.merged == false
    end

    test "clears merged_adapters list" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      {:ok, merged} = Layer.merge(layer)
      {:ok, unmerged} = Layer.unmerge(merged)

      assert unmerged.merged_adapters == []
    end

    test "returns error when not merged" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: nil}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      result = Layer.unmerge(layer)

      assert {:error, :not_merged} = result
    end

    test "also restores bias for non-feedforward" do
      base_layer = %{weight: Nx.broadcast(1.0, {4, 4}), bias: Nx.tensor([1.0, 1.0, 1.0, 1.0])}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", true)

      layer = put_in(layer.ia3_l["default"], Nx.tensor([2.0, 1.0, 0.5, 1.0]))

      {:ok, merged} = Layer.merge(layer)
      {:ok, unmerged} = Layer.unmerge(merged)

      assert_all_close(unmerged.base_layer.bias, Nx.tensor([1.0, 1.0, 1.0, 1.0]))
    end
  end

  describe "reset_ia3_parameters/2" do
    test "resets adapter weights to ones" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("default", false)

      # Verify it's not ones initially
      mean = Nx.to_number(Nx.mean(layer.ia3_l["default"]))
      assert mean != 1.0

      # Reset to ones
      layer = Layer.reset_ia3_parameters(layer, "default")

      # Now should be all ones
      assert Nx.to_number(Nx.mean(layer.ia3_l["default"])) == 1.0
    end
  end

  describe "set_adapter/2" do
    test "changes active adapter" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("adapter1", true)
        |> Layer.update_layer("adapter2", true)
        |> Layer.set_adapter("adapter2")

      assert layer.active_adapter == "adapter2"
    end
  end

  describe "delete_adapter/2" do
    test "removes adapter from ia3_l" do
      base_layer = %{weight: random_normal({64, 32})}

      layer =
        Layer.new(base_layer, false)
        |> Layer.update_layer("adapter1", true)
        |> Layer.update_layer("adapter2", true)
        |> Layer.delete_adapter("adapter1")

      refute Map.has_key?(layer.ia3_l, "adapter1")
      assert Map.has_key?(layer.ia3_l, "adapter2")
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
