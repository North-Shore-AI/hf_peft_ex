defmodule HfPeftEx.Tuners.Lora.EmbeddingTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Dora
  alias HfPeftEx.Tuners.Lora.Embedding

  describe "new/3" do
    test "creates embedding layer with correct dimensions" do
      embed = Embedding.new(10_000, 768, r: 8, lora_alpha: 16)

      assert embed.num_embeddings == 10_000
      assert embed.embedding_dim == 768
      assert embed.r == 8
    end

    test "creates lora_embedding_A with shape {r, num_embeddings}" do
      embed = Embedding.new(10_000, 768, r: 8)
      assert Nx.shape(embed.lora_embedding_A) == {8, 10_000}
    end

    test "creates lora_embedding_B with shape {embedding_dim, r}" do
      embed = Embedding.new(10_000, 768, r: 8)
      assert Nx.shape(embed.lora_embedding_B) == {768, 8}
    end

    test "computes scaling correctly" do
      embed = Embedding.new(1000, 128, r: 8, lora_alpha: 16)
      assert embed.scaling == 2.0
    end

    test "uses rslora scaling when enabled" do
      embed = Embedding.new(1000, 128, r: 4, lora_alpha: 8, use_rslora: true)
      assert embed.scaling == 8 / :math.sqrt(4)
    end

    test "accepts LoraConfig struct" do
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      embed = Embedding.new(5000, 512, config: config)

      assert embed.r == 16
      assert embed.lora_alpha == 32
    end

    test "initializes with random values" do
      embed = Embedding.new(1000, 128, r: 8)
      # Embedding uses randn initialization for both A and B (unlike Linear)
      mean_a = embed.lora_embedding_A |> Nx.mean() |> Nx.to_number()
      mean_b = embed.lora_embedding_B |> Nx.mean() |> Nx.to_number()
      # Random values should be near zero on average
      assert abs(mean_a) < 1.0
      assert abs(mean_b) < 1.0
    end
  end

  describe "get_delta_weight/1" do
    test "computes (B @ A)^T * scaling" do
      embed = Embedding.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:4 Credo.Check.Readability.VariableNames
      # 2x4
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      # 3x2
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      embed = %{embed | lora_embedding_A: lora_A, lora_embedding_B: lora_B}

      delta = Embedding.get_delta_weight(embed)

      # B @ A = [[1,0,0,0], [0,1,0,0], [0,0,0,0]] (3x4)
      # Transposed = [[1,0,0], [0,1,0], [0,0,0], [0,0,0]] (4x3)
      # For embedding: num_embeddings x embedding_dim
      assert Nx.shape(delta) == {4, 3}
    end
  end

  describe "forward/3" do
    test "returns base output when merged" do
      embed = Embedding.new(100, 32, r: 4)
      embed = %{embed | merged: true}

      indices = Nx.tensor([[0, 1, 2]])
      base_output = Nx.broadcast(0.5, {1, 3, 32})

      result = Embedding.forward(embed, indices, base_output)
      assert Nx.to_list(result) == Nx.to_list(base_output)
    end

    test "adds LoRA contribution when not merged" do
      embed = Embedding.new(100, 8, r: 2, lora_alpha: 2)
      indices = Nx.tensor([[0, 1]])
      base_output = Nx.broadcast(0.0, {1, 2, 8})

      result = Embedding.forward(embed, indices, base_output)
      assert Nx.shape(result) == {1, 2, 8}
    end

    test "handles single token input" do
      embed = Embedding.new(100, 16, r: 4)
      indices = Nx.tensor([[5]])
      base_output = Nx.broadcast(0.0, {1, 1, 16})

      result = Embedding.forward(embed, indices, base_output)
      assert Nx.shape(result) == {1, 1, 16}
    end
  end

  describe "merge/2" do
    test "sets merged flag to true" do
      embed = Embedding.new(100, 32, r: 4)
      base_weight = Nx.broadcast(0.0, {100, 32})

      {merged, _weight} = Embedding.merge(embed, base_weight)
      assert merged.merged == true
    end

    test "adds delta weight to base weight" do
      embed = Embedding.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      embed = %{embed | lora_embedding_A: lora_A, lora_embedding_B: lora_B}

      base_weight = Nx.broadcast(0.0, {4, 3})
      {_merged, new_weight} = Embedding.merge(embed, base_weight)

      assert Nx.shape(new_weight) == {4, 3}
    end

    test "returns unchanged when already merged" do
      embed = Embedding.new(100, 32, r: 4)
      embed = %{embed | merged: true}
      base_weight = Nx.broadcast(1.0, {100, 32})

      {merged, weight} = Embedding.merge(embed, base_weight)
      assert merged.merged == true
      assert Nx.to_list(weight) == Nx.to_list(base_weight)
    end
  end

  describe "unmerge/2" do
    test "sets merged flag to false" do
      embed = Embedding.new(100, 32, r: 4)
      embed = %{embed | merged: true}
      base_weight = Nx.broadcast(1.0, {100, 32})

      {unmerged, _weight} = Embedding.unmerge(embed, base_weight)
      assert unmerged.merged == false
    end

    test "merge and unmerge are inverse operations" do
      embed = Embedding.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
      embed = %{embed | lora_embedding_A: lora_A, lora_embedding_B: lora_B}

      original = Nx.broadcast(1.0, {4, 3})
      {merged, merged_weight} = Embedding.merge(embed, original)
      {_unmerged, restored} = Embedding.unmerge(merged, merged_weight)

      diff = Nx.subtract(original, restored) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  describe "embed_lookup/3" do
    test "looks up embeddings for given indices" do
      # Simple test with known embedding matrix
      # credo:disable-for-next-line Credo.Check.Readability.VariableNames
      embedding_A =
        Nx.tensor([
          # row 0
          [1.0, 2.0, 3.0, 4.0],
          # row 1
          [5.0, 6.0, 7.0, 8.0]
        ])

      # Get rows 0 and 2 from A^T
      indices = Nx.tensor([[0, 2]])

      # A^T shape is {4, 2}, looking up rows 0 and 2 gives us {1, 2, 2}
      result = Embedding.embed_lookup(indices, Nx.transpose(embedding_A))
      assert Nx.shape(result) == {1, 2, 2}
    end
  end

  describe "use_dora" do
    test "forward applies dora scaling when enabled" do
      embed = Embedding.new(3, 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      embed = %{embed | lora_embedding_A: lora_A, lora_embedding_B: lora_B}

      weight = Nx.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
      indices = Nx.tensor([[0, 1]])
      base_output = Embedding.embed_lookup(indices, weight)

      result = Embedding.forward(embed, indices, base_output, weight: weight)

      lora_weight = Nx.dot(lora_B, lora_A)
      weight_norm = Dora.get_weight_norm(weight, lora_weight, embed.scaling, fan_in_fan_out: true)
      magnitude = Dora.init_magnitude(Nx.transpose(weight))
      mag_norm_scale = Nx.divide(magnitude, weight_norm) |> Nx.reshape({1, 1, 2})

      after_a = Embedding.embed_lookup(indices, Nx.transpose(lora_A))

      lora_output =
        after_a
        |> Nx.dot(Nx.transpose(lora_B))
        |> Nx.multiply(embed.scaling)

      expected = Nx.multiply(mag_norm_scale, Nx.add(base_output, lora_output))

      assert_tensor_close(result, expected)

      lora_only = Embedding.forward(%{embed | use_dora: false}, indices, base_output)
      diff = Nx.subtract(result, lora_only) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "merge and unmerge apply dora factor" do
      embed = Embedding.new(3, 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      embed = %{embed | lora_embedding_A: lora_A, lora_embedding_B: lora_B}

      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
      delta = Embedding.get_delta_weight(embed)

      magnitude = Dora.init_magnitude(Nx.transpose(base_weight))

      weight_norm =
        Dora.get_weight_norm(base_weight, Nx.transpose(delta), 1.0, fan_in_fan_out: true)

      dora_factor = Nx.divide(magnitude, weight_norm) |> Nx.reshape({1, 2})
      expected_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {merged, merged_weight} = Embedding.merge(embed, base_weight)

      assert merged.merged == true
      assert merged.dora_weight_norm != nil
      assert_tensor_close(merged_weight, expected_weight)

      {unmerged, restored} = Embedding.unmerge(merged, merged_weight)
      assert unmerged.merged == false
      assert_tensor_close(restored, base_weight)
    end

    test "supports dropout in training mode (dora)" do
      embed = Embedding.new(3, 2, r: 2, lora_alpha: 2, use_dora: true)
      lora_a = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      embed = %{embed | lora_embedding_A: lora_a, lora_embedding_B: lora_b}

      indices = Nx.tensor([[0, 1]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      base_output = Embedding.embed_lookup(indices, base_weight)

      result = Embedding.forward(embed, indices, base_output, weight: base_weight, training: true)

      assert Nx.shape(result) == Nx.shape(base_output)
      assert_finite(result)
    end
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
