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

## Why PEFT?

| Benefit | Description |
|---------|-------------|
| **Memory Efficient** | Train 12B+ parameter models on consumer hardware |
| **Storage Efficient** | Adapter checkpoints are only a few MBs vs. GBs |
| **No Catastrophic Forgetting** | Base model weights remain frozen |
| **Composable** | Stack and switch between multiple adapters |

## Documentation

Full documentation is available at [HexDocs](https://hexdocs.pm/hf_peft_ex).

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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
