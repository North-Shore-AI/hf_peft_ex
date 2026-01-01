defmodule HfPeftEx.Tuners.PromptTuning.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.{Config, Embedding, Model}

  describe "new/2" do
    test "creates model with prompt encoder" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)

      model = Model.new(base_model, config)

      assert model.config == config
      assert model.prompt_encoder != nil
      assert %Embedding{} = model.prompt_encoder
    end

    test "stores base model reference" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)

      model = Model.new(base_model, config)

      assert model.base_model == base_model
    end

    test "extracts word embeddings from base model for sample_vocab" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          token_dim: 64,
          prompt_tuning_init: :sample_vocab
        )

      model = Model.new(base_model, config)

      # Should have created prompt encoder
      assert Nx.shape(model.prompt_encoder.embedding) == {10, 64}
    end

    test "accepts custom adapter name" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)

      model = Model.new(base_model, config, adapter_name: "my_prompt_adapter")

      assert model.adapter_name == "my_prompt_adapter"
    end

    test "defaults adapter name to 'default'" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)

      model = Model.new(base_model, config)

      assert model.adapter_name == "default"
    end
  end

  describe "prepare_inputs/3" do
    test "concatenates prompt embeddings with input embeddings" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      # batch=2, seq_len=10, dim=64
      key = Nx.Random.key(42)
      {input_embeds, _key} = Nx.Random.normal(key, shape: {2, 10, 64})

      {combined, _mask} = Model.prepare_inputs(model, input_embeds)

      # Should be [prompts (5) + input (10)] = 15
      assert Nx.shape(combined) == {2, 15, 64}
    end

    test "prepends prompt embeddings before input embeddings" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 3, token_dim: 4)
      model = Model.new(base_model, config)

      # Create known input embeddings
      input_embeds = Nx.broadcast(1.0, {1, 2, 4})

      {combined, _mask} = Model.prepare_inputs(model, input_embeds)

      # Last 2 positions should be the input (all 1.0)
      input_portion = combined[[0, 3..4, ..]]
      assert Nx.all_close(input_portion, Nx.broadcast(1.0, {2, 4})) |> Nx.to_number() == 1
    end

    test "updates attention mask for prompts" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      key = Nx.Random.key(42)
      {input_embeds, _key} = Nx.Random.normal(key, shape: {2, 10, 64})
      attention_mask = Nx.broadcast(1, {2, 10})

      {_combined, new_mask} =
        Model.prepare_inputs(model, input_embeds, attention_mask: attention_mask)

      assert Nx.shape(new_mask) == {2, 15}
      # First 5 positions should be 1 (prompts attend)
      first_five = new_mask[[0, 0..4]]
      assert Nx.to_number(Nx.sum(first_five)) == 5
    end

    test "creates attention mask if not provided" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 64)
      model = Model.new(base_model, config)

      key = Nx.Random.key(42)
      {input_embeds, _key} = Nx.Random.normal(key, shape: {2, 10, 64})

      {_combined, new_mask} = Model.prepare_inputs(model, input_embeds)

      # Should create all-ones mask for full sequence
      assert Nx.shape(new_mask) == {2, 15}
      # 2 * 15
      assert Nx.to_number(Nx.sum(new_mask)) == 30
    end

    test "preserves zeros in provided attention mask" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 3, token_dim: 64)
      model = Model.new(base_model, config)

      key = Nx.Random.key(42)
      {input_embeds, _key} = Nx.Random.normal(key, shape: {1, 5, 64})
      # Mask with some zeros (padding)
      attention_mask = Nx.tensor([[1, 1, 1, 0, 0]])

      {_combined, new_mask} =
        Model.prepare_inputs(model, input_embeds, attention_mask: attention_mask)

      # Total mask should be [1, 1, 1, 1, 1, 1, 0, 0] (3 prompt + original)
      assert Nx.shape(new_mask) == {1, 8}
      # Sum should be 6 (3 prompts + 3 valid input tokens)
      assert Nx.to_number(Nx.sum(new_mask)) == 6
    end
  end

  describe "get_trainable_params/1" do
    test "returns only prompt embedding parameters" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      model = Model.new(base_model, config)

      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "prompt_encoder.embedding")
      assert map_size(params) == 1
    end

    test "returned params have correct shape" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 20, token_dim: 128)
      model = Model.new(base_model, config)

      params = Model.get_trainable_params(model)

      assert Nx.shape(params["prompt_encoder.embedding"]) == {20, 128}
    end
  end

  describe "trainable_parameter_count/1" do
    test "returns correct parameter count" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      model = Model.new(base_model, config)

      count = Model.trainable_parameter_count(model)

      # 20 * 768 = 15,360
      assert count == 15_360
    end
  end

  describe "set_prompt_embedding/2" do
    test "updates prompt encoder embedding" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 4)
      model = Model.new(base_model, config)

      new_embedding = Nx.broadcast(2.0, {5, 4})
      updated_model = Model.set_prompt_embedding(model, new_embedding)

      assert Nx.all_close(updated_model.prompt_encoder.embedding, new_embedding) |> Nx.to_number() ==
               1
    end
  end

  describe "save_pretrained/2 and from_pretrained/2" do
    @tag :tmp_dir
    test "saves and loads prompt encoder", %{tmp_dir: dir} do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      model = Model.new(base_model, config)

      :ok = Model.save_pretrained(model, dir)

      # Verify files exist
      assert File.exists?(Path.join(dir, "adapter_config.json"))
      assert File.exists?(Path.join(dir, "adapter_model.bin"))

      # Load and verify
      {:ok, loaded} = Model.from_pretrained(base_model, dir)
      assert Nx.shape(loaded.prompt_encoder.embedding) == {10, 64}
      assert loaded.config.num_virtual_tokens == 10
    end

    @tag :tmp_dir
    test "loaded embedding matches saved embedding", %{tmp_dir: dir} do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 5, token_dim: 8)
      model = Model.new(base_model, config)

      # Set a known embedding
      known_embedding = Nx.iota({5, 8}) |> Nx.as_type(:f32)
      model = Model.set_prompt_embedding(model, known_embedding)

      :ok = Model.save_pretrained(model, dir)
      {:ok, loaded} = Model.from_pretrained(base_model, dir)

      assert Nx.all_close(loaded.prompt_encoder.embedding, known_embedding) |> Nx.to_number() == 1
    end
  end

  describe "prompt_learning?/1" do
    test "returns true" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      model = Model.new(base_model, config)

      assert Model.prompt_learning?(model) == true
    end
  end

  describe "get_peft_config/1" do
    test "returns the config" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 15, token_dim: 128)
      model = Model.new(base_model, config)

      retrieved_config = Model.get_peft_config(model)

      assert retrieved_config.num_virtual_tokens == 15
      assert retrieved_config.token_dim == 128
      assert retrieved_config.peft_type == :prompt_tuning
    end
  end

  # Helper function to create a mock base model
  defp create_mock_model do
    key = Nx.Random.key(123)
    {embeddings, _key} = Nx.Random.uniform(key, shape: {1000, 64})

    %{
      embeddings: embeddings,
      config: %{
        hidden_size: 64,
        vocab_size: 1000
      }
    }
  end
end
