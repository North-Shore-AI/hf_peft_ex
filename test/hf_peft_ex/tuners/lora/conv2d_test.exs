defmodule HfPeftEx.Tuners.Lora.Conv2dTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Conv2d

  describe "new/4" do
    test "creates lora matrices with correct shapes" do
      conv = Conv2d.new(3, 4, {2, 2}, r: 2)

      assert Nx.shape(conv.lora_A) == {2, 12}
      assert Nx.shape(conv.lora_B) == {4, 2}
    end

    test "accepts LoraConfig struct" do
      config = LoraConfig.new(r: 4, lora_alpha: 8)
      conv = Conv2d.new(3, 4, {2, 2}, config: config)

      assert conv.r == 4
      assert conv.scaling == 2.0
    end
  end

  describe "get_delta_weight/1" do
    test "returns delta weight with conv shape" do
      conv = Conv2d.new(2, 3, {2, 2}, r: 2, lora_alpha: 2)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      delta = Conv2d.get_delta_weight(conv)

      assert Nx.shape(delta) == {3, 2, 2, 2}
    end
  end

  describe "forward/4" do
    test "adds LoRA contribution when not merged" do
      conv = Conv2d.new(1, 2, {2, 2}, r: 2, lora_alpha: 2)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      x = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])

      base_weight =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 1.0]]],
          [[[0.5, 0.0], [0.0, 0.5]]]
        ])

      base_output = conv_out(x, base_weight, conv)
      lora_weight = lora_weight(conv, lora_b, lora_a)
      lora_output = conv_out(x, lora_weight, conv)

      result = Conv2d.forward(conv, x, base_output)

      expected = Nx.add(base_output, Nx.multiply(lora_output, conv.scaling))
      assert_tensor_close(result, expected)
    end

    test "applies dora scaling when enabled" do
      conv = Conv2d.new(1, 2, {2, 2}, r: 2, lora_alpha: 2, use_dora: true)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      x = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])

      base_weight =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 1.0]]],
          [[[0.5, 0.0], [0.0, 0.5]]]
        ])

      base_output = conv_out(x, base_weight, conv)
      lora_weight = lora_weight(conv, lora_b, lora_a)
      lora_output = conv_out(x, lora_weight, conv)

      magnitude = magnitude(base_weight)
      weight_norm = weight_norm(base_weight, lora_weight, conv.scaling)

      mag_norm_scale =
        Nx.divide(magnitude, weight_norm) |> Nx.reshape({1, conv.out_channels, 1, 1})

      expected =
        Nx.multiply(mag_norm_scale, Nx.add(base_output, Nx.multiply(lora_output, conv.scaling)))

      result = Conv2d.forward(conv, x, base_output, weight: base_weight)

      assert_tensor_close(result, expected)
    end

    test "supports dropout in training mode (lora)" do
      conv = Conv2d.new(1, 2, {2, 2}, r: 2, lora_alpha: 2, dropout: 0.5)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      x = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])

      base_weight =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 1.0]]],
          [[[0.5, 0.0], [0.0, 0.5]]]
        ])

      base_output = conv_out(x, base_weight, conv)

      result = Conv2d.forward(conv, x, base_output, training: true)

      assert Nx.shape(result) == Nx.shape(base_output)
      assert_finite(result)
    end

    test "supports dropout in training mode (dora)" do
      conv = Conv2d.new(1, 2, {2, 2}, r: 2, lora_alpha: 2, dropout: 0.5, use_dora: true)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      x = Nx.tensor([[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]])

      base_weight =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 1.0]]],
          [[[0.5, 0.0], [0.0, 0.5]]]
        ])

      base_output = conv_out(x, base_weight, conv)

      result = Conv2d.forward(conv, x, base_output, weight: base_weight, training: true)

      assert Nx.shape(result) == Nx.shape(base_output)
      assert_finite(result)
    end
  end

  describe "merge/2 and unmerge/2" do
    test "merge and unmerge are inverse operations" do
      conv = Conv2d.new(2, 2, {2, 2}, r: 2, lora_alpha: 2)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      base_weight = Nx.broadcast(1.0, {2, 2, 2, 2})
      {merged, merged_weight} = Conv2d.merge(conv, base_weight)
      {_unmerged, restored} = Conv2d.unmerge(merged, merged_weight)

      diff = Nx.subtract(base_weight, restored) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  describe "use_dora" do
    test "merge applies dora factor" do
      conv = Conv2d.new(1, 2, {2, 2}, r: 2, lora_alpha: 2, use_dora: true)

      lora_a =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      conv = %{conv | lora_A: lora_a, lora_B: lora_b}

      base_weight =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 1.0]]],
          [[[0.5, 0.0], [0.0, 0.5]]]
        ])

      delta = Conv2d.get_delta_weight(conv)
      lora_weight = lora_weight(conv, lora_b, lora_a)
      magnitude = magnitude(base_weight)
      weight_norm = weight_norm(base_weight, lora_weight, conv.scaling)
      dora_factor = Nx.divide(magnitude, weight_norm) |> Nx.reshape({conv.out_channels, 1, 1, 1})
      expected = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {merged, merged_weight} = Conv2d.merge(conv, base_weight)

      assert merged.dora_weight_norm != nil
      assert_tensor_close(merged_weight, expected)

      {_unmerged, restored} = Conv2d.unmerge(merged, merged_weight)
      assert_tensor_close(restored, base_weight)
    end
  end

  defp lora_weight(conv, lora_b, lora_a) do
    {k_h, k_w} = conv.kernel_size

    Nx.dot(lora_b, lora_a)
    |> Nx.reshape({conv.out_channels, conv.in_channels, k_h, k_w})
  end

  defp conv_out(x, weight, conv) do
    Nx.conv(x, weight,
      strides: conv.stride,
      padding: conv.padding,
      kernel_dilation: conv.dilation,
      feature_group_size: conv.groups
    )
  end

  defp magnitude(weight) do
    weight
    |> Nx.multiply(weight)
    |> Nx.sum(axes: [1, 2, 3])
    |> Nx.sqrt()
  end

  defp weight_norm(weight, lora_weight, scaling) do
    combined = Nx.add(weight, Nx.multiply(lora_weight, scaling))

    combined
    |> Nx.multiply(combined)
    |> Nx.sum(axes: [1, 2, 3])
    |> Nx.sqrt()
  end

  defp assert_tensor_close(left, right, tol \\ 1.0e-6) do
    diff = Nx.subtract(left, right) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
    assert diff < tol
  end

  defp assert_finite(tensor) do
    assert Nx.to_number(Nx.any(Nx.is_nan(tensor))) == 0
    assert Nx.to_number(Nx.any(Nx.is_infinity(tensor))) == 0
  end
end
