<p align="center">
  <img src="assets/hf_peft_ex.svg" alt="HF PEFT Elixir" width="200"/>
</p>

<h1 align="center">HF PEFT Ex</h1>

<p align="center">
  <strong>Elixir port of HuggingFace's PEFT (Parameter-Efficient Fine-Tuning) library</strong>
</p>

<p align="center">
  <a href="https://hex.pm/packages/hf_peft_ex"><img src="https://img.shields.io/hexpm/v/hf_peft_ex.svg?style=flat-square" alt="Hex.pm"></a>
  <a href="https://hexdocs.pm/hf_peft_ex"><img src="https://img.shields.io/badge/hex-docs-blue.svg?style=flat-square" alt="Docs"></a>
  <a href="https://github.com/North-Shore-AI/hf_peft_ex/actions"><img src="https://img.shields.io/github/actions/workflow/status/North-Shore-AI/hf_peft_ex/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/hexpm/l/hf_peft_ex.svg?style=flat-square" alt="License"></a>
  <a href="https://elixir-lang.org"><img src="https://img.shields.io/badge/elixir-%3E%3D%201.14-blueviolet.svg?style=flat-square" alt="Elixir"></a>
</p>

---

Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of large pretrained models by only fine-tuning a small number of parameters instead of the entire model. This Elixir library ports HuggingFace's [PEFT](https://github.com/huggingface/peft) to the BEAM ecosystem, providing native integration with [Nx](https://github.com/elixir-nx/nx) and [Axon](https://github.com/elixir-nx/axon).

## Features

- 🚀 **LoRA** - Low-Rank Adaptation for efficient fine-tuning
- 📊 **AdaLoRA** - Adaptive budget allocation for LoRA
- 🎯 **IA3** - Infused Adapter by Inhibiting and Amplifying Inner Activations
- 📝 **Prefix Tuning** - Prepending trainable prefixes to inputs
- 💬 **Prompt Tuning** - Learning soft prompts for task adaptation
- 🔧 **30+ PEFT methods** - Comprehensive coverage of state-of-the-art techniques

## Installation

Add `hf_peft_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:hf_peft_ex, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
alias HfPeftEx.PeftModel

# Configure LoRA
config = LoraConfig.new(
  r: 16,
  lora_alpha: 32,
  target_modules: ["q_proj", "v_proj"],
  task_type: :causal_lm
)

# Wrap your model with PEFT
peft_model = PeftModel.wrap(base_model, config)

# Train only the adapter parameters
# ... training loop ...

# Save the lightweight adapter
PeftModel.save(peft_model, "my-lora-adapter")
```

## Implemented PEFT Methods

### LoRA (Low-Rank Adaptation)

LoRA learns low-rank update matrices that are added to frozen pretrained weights.

```elixir
config = HfPeftEx.Tuners.Lora.Config.new(
  r: 16,
  lora_alpha: 32,
  target_modules: ["q_proj", "v_proj"],
  task_type: :causal_lm
)

# Create LoRA layer
layer = HfPeftEx.Tuners.Lora.Layer.new(
  in_features: 1024,
  out_features: 1024,
  config: config
)

# Forward pass adds LoRA contribution
result = HfPeftEx.Tuners.Lora.Layer.forward(layer, input, base_output)
```

**Parameter count:** `r × (in_features + out_features)` per layer.

### AdaLoRA (Adaptive LoRA)

AdaLoRA dynamically allocates rank budget across layers during training based on importance scores. Less important layers get pruned while important layers retain higher rank.

```elixir
config = HfPeftEx.Tuners.Adalora.Config.new(
  total_step: 10000,
  init_r: 12,
  target_r: 8,
  tinit: 200,
  tfinal: 200,
  delta_t: 10,
  beta1: 0.85,
  target_modules: ["q_proj", "v_proj"]
)

# Create AdaLoRA model with SVD parameterization
base_layers = %{
  "q_proj" => %{weight: q_weight, bias: nil},
  "v_proj" => %{weight: v_weight, bias: nil}
}
model = HfPeftEx.Tuners.Adalora.Model.new(config, base_layers)

# In training loop, after backward pass
gradients = collect_gradients(model)
{model, masks} = HfPeftEx.Tuners.Adalora.Model.update_and_allocate(model, gradients, step)

# Add orthogonal regularization to loss
orth_loss = HfPeftEx.Tuners.Adalora.Model.get_orthogonal_loss(model)
total_loss = loss + config.orth_reg_weight * orth_loss
```

**Key features:**
- SVD parameterization with prunable singular values
- Dynamic rank allocation based on importance scores
- EMA smoothing of importance with uncertainty quantification
- Cubic budget schedule from `init_r` to `target_r`
- Orthogonal regularization for stability

**Three phases:**
1. **Warmup (tinit)**: Pre-training without rank pruning
2. **Budgeting**: Rank decreases from `init_r` to `target_r`
3. **Final (tfinal)**: Fine-tuning with frozen ranks

### IA3 (Infused Adapter by Inhibiting and Amplifying Activations)

IA3 learns multiplicative scaling vectors for layer activations. Much simpler than LoRA with minimal parameters.

```elixir
config = HfPeftEx.Tuners.IA3.Config.new(
  target_modules: ["q_proj", "v_proj", "down_proj"],
  feedforward_modules: ["down_proj"],
  init_ia3_weights: true
)

# Create IA3 linear layer
linear = HfPeftEx.Tuners.IA3.Linear.new(1024, 512, config: config)

# Forward pass scales output
output = HfPeftEx.Tuners.IA3.Linear.forward(linear, input, base_output)

# Merge for inference
{merged_linear, new_weight} = HfPeftEx.Tuners.IA3.Linear.merge(linear, base_weight)
```

**Parameter count:** Only `d` parameters per layer (the scaling vector).

### Prefix Tuning

Prefix Tuning prepends trainable prefix tokens to the keys and values of each attention layer. Unlike Prompt Tuning which only modifies the input embeddings, Prefix Tuning modifies all transformer layers.

```elixir
config = HfPeftEx.Tuners.PrefixTuning.Config.new(
  num_virtual_tokens: 20,
  num_layers: 12,
  token_dim: 768,
  num_attention_heads: 12,
  prefix_projection: true,
  encoder_hidden_size: 512
)

# Create prefix tuning model
model = HfPeftEx.Tuners.PrefixTuning.Model.new(base_model, config)

# Get past_key_values for attention layers
past_key_values = HfPeftEx.Tuners.PrefixTuning.Model.get_past_key_values(model, batch_size)

# Extend attention mask for prefix tokens
new_mask = HfPeftEx.Tuners.PrefixTuning.Model.prepare_attention_mask(model, batch_size, attention_mask)

# Get trainable parameters
params = HfPeftEx.Tuners.PrefixTuning.Model.get_trainable_params(model)
```

**Options:**
- `prefix_projection: false` - Direct embedding (default)
- `prefix_projection: true` - Use MLP for reparameterization (requires `encoder_hidden_size`)

**Parameter count:**
- Without projection: `num_virtual_tokens × num_layers × 2 × token_dim`
- With projection: Reduced through bottleneck MLP

### Prompt Tuning

Prompt Tuning adds learnable virtual tokens (soft prompts) to the input. Only the prompt embeddings are trained while the base model remains frozen.

```elixir
config = HfPeftEx.Tuners.PromptTuning.Config.new(
  num_virtual_tokens: 20,
  token_dim: 768,
  prompt_tuning_init: :random
)

# Create prompt tuning model
model = HfPeftEx.Tuners.PromptTuning.Model.new(base_model, config)

# Prepare inputs with prompts prepended
{combined_embeds, attention_mask} = HfPeftEx.Tuners.PromptTuning.Model.prepare_inputs(
  model,
  input_embeds,
  attention_mask: mask
)

# Get trainable parameters for optimization
params = HfPeftEx.Tuners.PromptTuning.Model.get_trainable_params(model)

# Save and load adapter
:ok = HfPeftEx.Tuners.PromptTuning.Model.save_pretrained(model, "my-prompt-adapter")
{:ok, loaded} = HfPeftEx.Tuners.PromptTuning.Model.from_pretrained(base_model, "my-prompt-adapter")
```

**Initialization options:**
- `:random` - Random continuous vectors (default)
- `:sample_vocab` - Sample from vocabulary embeddings
- `:text` - Initialize from tokenized text

**Parameter count:** `num_virtual_tokens × token_dim` parameters.

## Why PEFT?

| Benefit | Description |
|---------|-------------|
| **Memory Efficient** | Train 12B+ parameter models on consumer hardware |
| **Storage Efficient** | Adapter checkpoints are only a few MBs vs. GBs |
| **No Catastrophic Forgetting** | Base model weights remain frozen |
| **Composable** | Stack and switch between multiple adapters |

## Core Utilities

### Mapping Registry

The `HfPeftEx.Mapping` module provides the registry for PEFT type-to-implementation mappings:

```elixir
# Get config class for PEFT type
HfPeftEx.Mapping.get_config_class(:lora)
#=> HfPeftEx.Tuners.Lora.Config

# Get tuner/model class for PEFT type
HfPeftEx.Mapping.get_tuner_class(:lora)
#=> HfPeftEx.Tuners.Lora.Model

# Create config from dict (e.g., loaded from JSON)
{:ok, config} = HfPeftEx.Mapping.get_peft_config(%{
  "peft_type" => "lora",
  "r" => 8,
  "lora_alpha" => 16
})
```

### Target Module Constants

The `HfPeftEx.Constants` module provides default target modules for common model architectures:

```elixir
# Get default LoRA target modules for model type
HfPeftEx.Constants.get_lora_target_modules("llama")
#=> ["q_proj", "v_proj"]

HfPeftEx.Constants.get_lora_target_modules("gpt2")
#=> ["c_attn"]

# Get IA3 target modules
HfPeftEx.Constants.get_ia3_target_modules("llama")
#=> ["k_proj", "v_proj", "down_proj"]

# Generic lookup by PEFT type
HfPeftEx.Constants.get_target_modules(:lora, "bert")
#=> ["query", "value"]

# File name constants
HfPeftEx.Constants.config_name()            #=> "adapter_config.json"
HfPeftEx.Constants.safetensors_weights_name() #=> "adapter_model.safetensors"
```

### Save and Load

The `HfPeftEx.Utils.SaveAndLoad` module handles adapter weight persistence:

```elixir
alias HfPeftEx.Utils.SaveAndLoad

# Extract adapter weights from model
{:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)
{:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model, "custom_adapter")

# Save adapter weights to file
:ok = SaveAndLoad.save_peft_weights(state_dict, "adapter_model.nx")

# Load adapter weights from file
{:ok, weights} = SaveAndLoad.load_peft_weights("adapter_model.nx")

# Load weights into model
{:ok, model} = SaveAndLoad.set_peft_model_state_dict(model, weights)
{:ok, model} = SaveAndLoad.set_peft_model_state_dict(model, weights, "new_adapter")
```

## Documentation

Full documentation is available at [HexDocs](https://hexdocs.pm/hf_peft_ex).

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/North-Shore-AI/hf_peft_ex).

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

This library is a port of [HuggingFace PEFT](https://github.com/huggingface/peft). All credit for the original algorithms and implementations goes to the HuggingFace team and the original paper authors.

```bibtex
@Misc{peft,
  title =        {{PEFT}: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan and Marian Tietz},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```
