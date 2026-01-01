defmodule HfPeftEx.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/hf_peft_ex"

  def project do
    [
      app: :hf_peft_ex,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      description: description(),
      package: package(),
      name: "HfPeftEx",
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:jason, "~> 1.4"},
      {:nx, "~> 0.7"},
      {:axon, "~> 0.6"},
      {:ex_doc, "~> 0.38", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp description do
    """
    Elixir port of HuggingFace PEFT (Parameter-Efficient Fine-Tuning) library.
    Provides LoRA and other efficient fine-tuning adapters for Nx/Axon models.
    """
  end

  defp docs do
    [
      main: "readme",
      name: "HfPeftEx",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      assets: %{"assets" => "assets"},
      logo: "assets/hf_peft_ex.svg",
      extras: ["README.md"]
    ]
  end

  defp package do
    [
      name: "hf_peft_ex",
      files: ~w(lib mix.exs README.md LICENSE assets),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "HuggingFace PEFT" => "https://github.com/huggingface/peft"
      }
    ]
  end
end
