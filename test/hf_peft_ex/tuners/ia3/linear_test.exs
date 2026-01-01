defmodule HfPeftEx.Tuners.IA3.LinearTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.Config, as: IA3Config
  alias HfPeftEx.Tuners.IA3.Linear

  describe "new/3" do
    test "creates linear layer with correct dimensions" do
      linear = Linear.new(1024, 512)

      assert linear.in_features == 1024
      assert linear.out_features == 512
      assert linear.is_feedforward == false
    end

    test "creates ia3_l with correct shape for non-feedforward" do
      linear = Linear.new(1024, 512)
      assert Nx.shape(linear.ia3_l) == {512}
    end

    test "creates ia3_l with correct shape for feedforward" do
      linear = Linear.new(1024, 512, is_feedforward: true)
      assert Nx.shape(linear.ia3_l) == {1024}
    end

    test "initializes ia3_l to ones by default" do
      linear = Linear.new(64, 32)
      assert Nx.to_number(Nx.mean(linear.ia3_l)) == 1.0
      assert Nx.to_number(Nx.reduce_min(linear.ia3_l)) == 1.0
      assert Nx.to_number(Nx.reduce_max(linear.ia3_l)) == 1.0
    end

    test "accepts init_ia3_weights option" do
      linear = Linear.new(64, 32, init_ia3_weights: false)
      mean = Nx.to_number(Nx.mean(linear.ia3_l))
      # Random init - mean should not be 1.0
      assert mean != 1.0
    end

    test "defaults fan_in_fan_out to false" do
      linear = Linear.new(256, 128)
      assert linear.fan_in_fan_out == false
    end

    test "accepts fan_in_fan_out option" do
      linear = Linear.new(256, 128, fan_in_fan_out: true)
      assert linear.fan_in_fan_out == true
    end

    test "accepts IA3Config struct" do
      config = IA3Config.new(init_ia3_weights: false, fan_in_fan_out: true)
      linear = Linear.new(512, 256, config: config)

      assert linear.fan_in_fan_out == true
      # Random init from config
      mean = Nx.to_number(Nx.mean(linear.ia3_l))
      assert mean != 1.0
    end

    test "explicit options override config options" do
      config = IA3Config.new(init_ia3_weights: true)
      linear = Linear.new(64, 32, config: config, init_ia3_weights: false)

      # Explicit opts override config, so init_ia3_weights: false wins
      mean = Nx.to_number(Nx.mean(linear.ia3_l))
      # Random init - mean should not be 1.0
      assert mean != 1.0
    end

    test "stores is_feedforward flag" do
      linear = Linear.new(64, 32, is_feedforward: true)
      assert linear.is_feedforward == true
    end
  end

  describe "forward/3" do
    test "returns identity when ia3_l is ones and weight is identity" do
      linear = Linear.new(4, 4)

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      # Identity operation
      base_output = x

      result = Linear.forward(linear, x, base_output)
      assert_all_close(result, base_output)
    end

    test "scales output by ia3_l" do
      linear = Linear.new(4, 4)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 2.0, 2.0, 2.0])}

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      base_output = x

      result = Linear.forward(linear, x, base_output)
      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0]])
      assert_all_close(result, expected)
    end

    test "applies different scaling per output dimension" do
      linear = Linear.new(4, 4)
      linear = %{linear | ia3_l: Nx.tensor([1.0, 2.0, 0.5, 3.0])}

      x = Nx.tensor([[2.0, 2.0, 2.0, 2.0]])
      base_output = x

      result = Linear.forward(linear, x, base_output)
      expected = Nx.tensor([[2.0, 4.0, 1.0, 6.0]])
      assert_all_close(result, expected)
    end

    test "returns base output when merged" do
      linear = Linear.new(4, 4)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 2.0, 2.0, 2.0]), merged: true}

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      base_output = x

      result = Linear.forward(linear, x, base_output)
      assert_all_close(result, base_output)
    end

    test "handles batch input" do
      linear = Linear.new(4, 4)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 2.0, 2.0, 2.0])}

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]])
      base_output = x

      result = Linear.forward(linear, x, base_output)
      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
      assert_all_close(result, expected)
    end
  end

  describe "get_scaling_vector/1" do
    test "returns ia3_l tensor" do
      linear = Linear.new(64, 32)
      scaling = Linear.get_scaling_vector(linear)
      assert Nx.shape(scaling) == {32}
      assert_all_close(scaling, Nx.broadcast(1.0, {32}))
    end

    test "returns correct shape for feedforward" do
      linear = Linear.new(64, 32, is_feedforward: true)
      scaling = Linear.get_scaling_vector(linear)
      assert Nx.shape(scaling) == {64}
    end
  end

  describe "merge/2" do
    test "sets merged flag to true" do
      linear = Linear.new(4, 3)
      base_weight = Nx.broadcast(1.0, {3, 4})

      {merged, _weight} = Linear.merge(linear, base_weight)
      assert merged.merged == true
    end

    test "scales rows of weight for non-feedforward" do
      linear = Linear.new(4, 3)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5])}

      base_weight = Nx.broadcast(1.0, {3, 4})
      {_merged, new_weight} = Linear.merge(linear, base_weight)

      expected =
        Nx.tensor([
          [2.0, 2.0, 2.0, 2.0],
          [1.0, 1.0, 1.0, 1.0],
          [0.5, 0.5, 0.5, 0.5]
        ])

      assert_all_close(new_weight, expected)
    end

    test "scales columns of weight for feedforward" do
      linear = Linear.new(4, 3, is_feedforward: true)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5, 1.0])}

      base_weight = Nx.broadcast(1.0, {3, 4})
      {_merged, new_weight} = Linear.merge(linear, base_weight)

      expected =
        Nx.tensor([
          [2.0, 1.0, 0.5, 1.0],
          [2.0, 1.0, 0.5, 1.0],
          [2.0, 1.0, 0.5, 1.0]
        ])

      assert_all_close(new_weight, expected)
    end

    test "handles fan_in_fan_out transposition" do
      linear = Linear.new(4, 3, fan_in_fan_out: true)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5])}

      # For fan_in_fan_out, base_weight is {in, out} = {4, 3}
      base_weight = Nx.broadcast(1.0, {4, 3})
      {_merged, new_weight} = Linear.merge(linear, base_weight)

      # Columns should be scaled
      expected =
        Nx.tensor([
          [2.0, 1.0, 0.5],
          [2.0, 1.0, 0.5],
          [2.0, 1.0, 0.5],
          [2.0, 1.0, 0.5]
        ])

      assert_all_close(new_weight, expected)
    end

    test "returns unchanged when already merged" do
      linear = Linear.new(4, 3)
      linear = %{linear | merged: true}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {merged, weight} = Linear.merge(linear, base_weight)
      assert merged.merged == true
      assert_all_close(weight, base_weight)
    end
  end

  describe "unmerge/2" do
    test "sets merged flag to false" do
      linear = Linear.new(4, 3)
      linear = %{linear | merged: true}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {unmerged, _weight} = Linear.unmerge(linear, base_weight)
      assert unmerged.merged == false
    end

    test "merge and unmerge are inverse operations" do
      linear = Linear.new(4, 3)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5])}

      original = Nx.broadcast(1.0, {3, 4})
      {merged, merged_weight} = Linear.merge(linear, original)
      {_unmerged, restored} = Linear.unmerge(merged, merged_weight)

      assert_all_close(restored, original)
    end

    test "merge and unmerge work for feedforward" do
      linear = Linear.new(4, 3, is_feedforward: true)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5, 3.0])}

      original = Nx.broadcast(1.0, {3, 4})
      {merged, merged_weight} = Linear.merge(linear, original)
      {_unmerged, restored} = Linear.unmerge(merged, merged_weight)

      assert_all_close(restored, original)
    end

    test "merge and unmerge work with fan_in_fan_out" do
      linear = Linear.new(4, 3, fan_in_fan_out: true)
      linear = %{linear | ia3_l: Nx.tensor([2.0, 1.0, 0.5])}

      original = Nx.broadcast(1.0, {4, 3})
      {merged, merged_weight} = Linear.merge(linear, original)
      {_unmerged, restored} = Linear.unmerge(merged, merged_weight)

      assert_all_close(restored, original)
    end

    test "returns unchanged when already unmerged" do
      linear = Linear.new(4, 3)
      linear = %{linear | merged: false}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {unmerged, weight} = Linear.unmerge(linear, base_weight)
      assert unmerged.merged == false
      assert_all_close(weight, base_weight)
    end
  end

  describe "reset_ia3_parameters/1" do
    test "resets ia3_l to ones" do
      linear = Linear.new(64, 32, init_ia3_weights: false)
      mean = Nx.to_number(Nx.mean(linear.ia3_l))
      assert mean != 1.0

      linear = Linear.reset_ia3_parameters(linear)
      assert Nx.to_number(Nx.mean(linear.ia3_l)) == 1.0
    end
  end

  describe "trainable_params/1" do
    test "returns size of ia3_l" do
      linear = Linear.new(1024, 512)
      assert Linear.trainable_params(linear) == 512
    end

    test "returns in_features for feedforward" do
      linear = Linear.new(1024, 512, is_feedforward: true)
      assert Linear.trainable_params(linear) == 1024
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
