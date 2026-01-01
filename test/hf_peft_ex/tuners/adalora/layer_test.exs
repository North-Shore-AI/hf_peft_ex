defmodule HfPeftEx.Tuners.Adalora.LayerTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.Layer

  defp random_normal(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  describe "new/2" do
    test "creates layer with correct dimensions" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}
      layer = Layer.new(base_layer)

      assert layer.in_features == 32
      assert layer.out_features == 64
    end

    test "extracts dimensions with fan_in_fan_out" do
      # When fan_in_fan_out, weight is {in, out} instead of {out, in}
      base_layer = %{weight: random_normal({32, 64}), bias: nil}
      layer = Layer.new(base_layer, fan_in_fan_out: true)

      assert layer.in_features == 32
      assert layer.out_features == 64
    end

    test "stores base layer reference" do
      base_layer = %{weight: Nx.eye(4), bias: nil}
      layer = Layer.new(base_layer)

      assert layer.base_layer == base_layer
    end
  end

  describe "update_layer/4" do
    test "initializes SVD parameterization" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
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
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)

      # E should be initialized to ones (not zeros like Python reset_lora_parameters)
      assert Nx.to_number(Nx.mean(layer.lora_e["default"])) == 1.0
    end

    test "sets scaling based on lora_alpha" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)

      # scaling = lora_alpha / r = 16 / 8 = 2.0
      assert layer.scaling["default"] == 2.0
    end

    test "supports multiple adapters" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)
        |> Layer.update_layer("other", 4, 8)

      assert Map.has_key?(layer.lora_a, "default")
      assert Map.has_key?(layer.lora_a, "other")
      assert Nx.shape(layer.lora_a["other"]) == {4, 32}
    end
  end

  describe "forward/2" do
    test "computes output with base layer and lora delta" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)

      input = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      output = Layer.forward(layer, input)

      # Output shape should match base layer output
      assert Nx.shape(output) == {1, 4}
    end

    test "returns base output when disabled" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)
        |> Map.put(:disable_adapters, true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      # Should equal base output (x @ W^T = x @ I = x)
      assert_all_close(output, input)
    end

    test "returns base output when merged" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)
        |> Map.put(:merged, true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      assert_all_close(output, input)
    end

    test "adds bias when present" do
      base_layer = %{weight: Nx.eye(4), bias: Nx.tensor([1.0, 1.0, 1.0, 1.0])}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)
        |> Map.put(:disable_adapters, true)

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      output = Layer.forward(layer, input)

      expected = Nx.tensor([[2.0, 3.0, 4.0, 5.0]])
      assert_all_close(output, expected)
    end
  end

  describe "get_delta_weight/2" do
    test "computes B @ diag(E) @ A scaled" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)

      delta = Layer.get_delta_weight(layer, "default")

      assert Nx.shape(delta) == {64, 32}
    end

    test "delta is zero when lora_b is zero" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)

      # lora_b is initialized to zeros, so delta should be zero
      delta = Layer.get_delta_weight(layer, "default")

      assert_all_close(delta, Nx.broadcast(0.0, {64, 32}))
    end
  end

  describe "apply_mask/3" do
    test "zeros out pruned singular values" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 8, 16)

      # Mask: keep first 4, prune last 4
      mask =
        Nx.concatenate([Nx.broadcast(1.0, {4, 1}), Nx.broadcast(0.0, {4, 1})], axis: 0)

      masked_layer = Layer.apply_mask(layer, "default", mask)

      # Check rank updated
      assert masked_layer.ranknum["default"] == 4

      # Check E is masked
      e = masked_layer.lora_e["default"]
      # First 4 should be 1.0, last 4 should be 0.0
      assert Nx.to_number(e[0][0]) == 1.0
      assert Nx.to_number(e[3][0]) == 1.0
      assert Nx.to_number(e[4][0]) == 0.0
      assert Nx.to_number(e[7][0]) == 0.0
    end

    test "handles all zeros mask" do
      base_layer = %{weight: random_normal({64, 32}), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 4, 8)

      mask = Nx.broadcast(0.0, {4, 1})
      masked_layer = Layer.apply_mask(layer, "default", mask)

      assert masked_layer.ranknum["default"] == 0
    end
  end

  describe "merge/2" do
    test "adds delta weight to base weight" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)

      # Set lora_b to non-zero for actual delta
      layer =
        put_in(
          layer.lora_b["default"],
          Nx.broadcast(0.1, {4, 2})
        )

      base_weight = Nx.eye(4)
      {merged_layer, new_weight} = Layer.merge(layer, "default", base_weight)

      assert merged_layer.merged == true
      assert Nx.shape(new_weight) == {4, 4}
    end

    test "returns unchanged when already merged" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)
        |> Map.put(:merged, true)

      base_weight = Nx.eye(4)
      {merged_layer, new_weight} = Layer.merge(layer, "default", base_weight)

      assert merged_layer.merged == true
      assert_all_close(new_weight, base_weight)
    end
  end

  describe "unmerge/2" do
    test "subtracts delta weight from base weight" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)

      # Set lora_b to non-zero
      layer =
        put_in(
          layer.lora_b["default"],
          Nx.broadcast(0.1, {4, 2})
        )

      # First merge
      base_weight = Nx.eye(4)
      {merged_layer, merged_weight} = Layer.merge(layer, "default", base_weight)

      # Then unmerge
      {unmerged_layer, restored_weight} =
        Layer.unmerge(merged_layer, "default", merged_weight)

      assert unmerged_layer.merged == false
      assert_all_close(restored_weight, base_weight, atol: 1.0e-5)
    end

    test "returns unchanged when not merged" do
      base_layer = %{weight: Nx.eye(4), bias: nil}

      layer =
        base_layer
        |> Layer.new()
        |> Layer.update_layer("default", 2, 2)

      base_weight = Nx.eye(4)
      {unmerged_layer, new_weight} = Layer.unmerge(layer, "default", base_weight)

      assert unmerged_layer.merged == false
      assert_all_close(new_weight, base_weight)
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
