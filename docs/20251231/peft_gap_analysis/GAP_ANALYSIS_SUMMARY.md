# PEFT Library Gap Analysis: Python vs Elixir Port

## Executive Summary

This document provides a comprehensive one-way gap analysis identifying all functionality in the HuggingFace PEFT Python library that remains to be ported to the Elixir implementation (`hf_peft_ex`).

**Current State:**
- Python PEFT: 31 PEFT methods, 162 Python files, 32+ configuration classes
- Elixir Port: 13 modules, 1 PEFT method (LoRA with DoRA), 149 passing tests

---

## Gap Categories

| Category | Python PEFT | Elixir Port | Gap |
|----------|-------------|-------------|-----|
| PEFT Methods (Tuners) | 31 | 1 (LoRA) | 30 methods |
| Configuration Classes | 32+ | 2 (Config, LoraConfig) | 30+ configs |
| Utility Modules | 11 | 0 | 11 modules |
| Core Infrastructure | 8 | 3 | 5 modules |
| Task-Specific Models | 6 | 0 | 6 models |
| Quantization Backends | 8 | N/A | Not applicable |

---

## 1. PEFT Methods (Tuners) - 30 Methods Missing

### Priority 1: High-Value Methods
| Method | Complexity | Value | Files to Port |
|--------|------------|-------|---------------|
| AdaLoRA | Medium | High | config.py, layer.py, model.py |
| IA3 | Low | High | config.py, layer.py, model.py |
| Prefix Tuning | Medium | High | config.py, model.py |
| Prompt Tuning | Low | High | config.py, model.py |
| OFT | Medium | Medium | config.py, layer.py, model.py |
| BOFT | Medium | Medium | config.py, layer.py, model.py |

### Priority 2: LyCORIS Family
| Method | Complexity | Value | Files to Port |
|--------|------------|-------|---------------|
| LoHa | Medium | Medium | config.py, layer.py, model.py |
| LoKr | Medium | Medium | config.py, layer.py, model.py |

### Priority 3: Advanced Methods
| Method | Complexity | Value | Files to Port |
|--------|------------|-------|---------------|
| VeRA | High | Medium | config.py, layer.py, model.py |
| VBLoRA | High | Medium | config.py, layer.py, model.py |
| XLoRA | High | Medium | config.py, classifier.py, model.py |
| Poly | Medium | Low | config.py, layer.py, model.py |
| FourierFT | High | Low | config.py, layer.py, model.py |
| WaveFT | High | Low | config.py, layer.py, model.py |

### Priority 4: Specialized Methods
| Method | Files | Notes |
|--------|-------|-------|
| P-Tuning | 3 | MLP/LSTM encoder |
| Multitask Prompt Tuning | 3 | Multi-task learning |
| CPT | 3 | Context-aware prompts |
| Adaption Prompt | 3 | Llama-specific |
| LN Tuning | 3 | LayerNorm only |
| Trainable Tokens | 3 | Token-level |
| AdaptionPrompt | 3 | Attention-based |
| RandLoRA | 4 | Random projections |
| DeLoRA | 3 | Bounded updates |
| GraLoRA | 3 | Gradient-aware |
| HRA | 3 | Gram-Schmidt |
| Bone | 3 | Block orthogonal |
| MiSS | 3 | Sharded scaling |
| RoAd | 3 | Rotation-based |
| ShiRA | 4 | Sparse high-rank |
| C3A | 3 | Block-diagonal |
| OSF | 3 | Orthogonal subspace |

---

## 2. Core Infrastructure - 5 Modules Missing

### 2.1 Mapping System (`mapping.py`)
**Status:** Not started
**Purpose:** Type-to-class registry for PEFT methods
**Key Functions:**
- `PEFT_TYPE_TO_CONFIG_MAPPING`
- `PEFT_TYPE_TO_TUNER_MAPPING`
- `register_peft_method/1`
- `get_peft_config/1`
- `inject_adapter_in_model/3`

### 2.2 Auto Model Loading (`auto.py`)
**Status:** Not started
**Purpose:** Automatic model/config loading from Hub
**Key Classes:**
- `AutoPeftModel`
- `AutoPeftModelForCausalLM`
- `AutoPeftModelForSeq2SeqLM`
- `AutoPeftModelForSequenceClassification`
- `AutoPeftModelForTokenClassification`
- `AutoPeftModelForQuestionAnswering`
- `AutoPeftModelForFeatureExtraction`

### 2.3 Mixed Model Support (`mixed_model.py`)
**Status:** Not started
**Purpose:** Combine multiple adapter types
**Key Features:**
- Multiple adapter type combinations
- Inference-only mode
- Adapter stacking

### 2.4 Helpers (`helpers.py`)
**Status:** Not started
**Purpose:** Model signature utilities
**Key Functions:**
- `update_forward_signature/1`
- `update_generate_signature/1`
- `check_if_peft_model/1`
- `rescale_adapter_scale/3` (context manager)
- `disable_input_dtype_casting/1`

### 2.5 Tuners Utils (`tuners/tuners_utils.py`)
**Status:** Partial (needs enhancement)
**Purpose:** Base tuner infrastructure
**Key Classes:**
- `BaseTuner` (abstract base)
- `BaseTunerLayer` (mixin)
- `LycorisTuner` (LyCORIS base)
- `LycorisLayer` (LyCORIS mixin)
**Key Functions:**
- `onload_layer/1` (context manager)
- Target module matching logic

---

## 3. Utility Modules - 11 Modules Missing

### 3.1 Save and Load (`utils/save_and_load.py`)
**Status:** Not started
**Purpose:** Adapter state dict operations
**Key Functions:**
- `get_peft_model_state_dict/3`
- `set_peft_model_state_dict/3`
- `load_peft_weights/3`
**Lines:** 724

### 3.2 Merge Utils (`utils/merge_utils.py`)
**Status:** Not started
**Purpose:** Multi-adapter merging strategies
**Key Functions:**
- `task_arithmetic/3`
- `magnitude_prune/4`
- `ties/4`
- `dare_linear/4`
- `dare_ties/4`
**Lines:** 268

### 3.3 Other Utilities (`utils/other.py`)
**Status:** Not started
**Purpose:** Core utility functions
**Key Functions:**
- `prepare_model_for_kbit_training/2`
- `_get_submodules/2`
- `_set_trainable/3`
- `_freeze_adapter/2`
- `match_target_against_key/2`
**Key Classes:**
- `ModulesToSaveWrapper`
- `AuxiliaryTrainingWrapper`
- `TrainableTokensWrapper`
**Lines:** 1648

### 3.4 Integrations (`utils/integrations.py`)
**Status:** Not started
**Purpose:** Framework integration helpers
**Key Functions:**
- `dequantize_module_weight/1`
- `get_layer_device_map/1`
- `map_cache_to_layer_device_map/2`
- `init_empty_weights/0` (context manager)
**Lines:** 291

### 3.5 Hotswap (`utils/hotswap.py`)
**Status:** Not started
**Purpose:** Runtime adapter switching
**Key Functions:**
- `prepare_model_for_compiled_hotswap/1`
- `hotswap_adapter/3`
- `check_hotswap_configs_compatible/2`
**Lines:** 630

### 3.6 LoftQ Utils (`utils/loftq_utils.py`)
**Status:** Not started
**Purpose:** Quantization-aware LoRA initialization
**Key Classes:**
- `NFQuantizer`
**Key Functions:**
- `replace_lora_weights_loftq/2`
**Lines:** 410

### 3.7 Incremental PCA (`utils/incremental_pca.py`)
**Status:** Not started
**Purpose:** GPU-accelerated incremental PCA
**Key Classes:**
- `IncrementalPCA`
**Lines:** 338

### 3.8 Constants (`utils/constants.py`)
**Status:** Not started
**Purpose:** Model-to-target-module mappings
**Key Constants:**
- `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`
- `TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING`
- `EMBEDDING_LAYER_NAMES`
- And 15+ more mappings
**Lines:** 362

### 3.9 PEFT Types (`utils/peft_types.py`)
**Status:** Partial (enums exist, need TaskType details)
**Purpose:** Type definitions
**Existing:** `PeftType` enum with 31 types, `TaskType` with 6 types
**Missing:** Helper functions, type validation

### 3.10 Import Utils (`import_utils.py`)
**Status:** Not started
**Purpose:** Optional dependency checking
**Key Functions:**
- `is_bnb_available/0`
- `is_gptq_available/0`
- Various `is_*_available` functions

### 3.11 Functional (`functional.py`)
**Status:** Not started
**Purpose:** Functional API utilities
**Note:** Minimal content, low priority

---

## 4. Task-Specific Models - 6 Models Missing

### Currently in Python PEFT:
1. `PeftModelForSequenceClassification`
2. `PeftModelForCausalLM`
3. `PeftModelForSeq2SeqLM`
4. `PeftModelForTokenClassification`
5. `PeftModelForQuestionAnswering`
6. `PeftModelForFeatureExtraction`

### Elixir Port Status:
- Basic `PeftModel` wrapper exists
- No task-specific subclasses
- Forward pass customizations not implemented

---

## 5. Advanced LoRA Features - Partial Implementation

### Currently Implemented:
- Basic LoRA (Linear, Embedding, Conv1d, Conv2d)
- DoRA (Weight-Decomposed LoRA)
- Scaling (standard and rank-stabilized)
- Merge/unmerge operations

### Missing:
- `use_qalora` (Quantization-Aware LoRA)
- `trainable_token_indices` (selective token training)
- `use_bdlora` (Block-Diagonal LoRA)
- `arrow_config` (Arrow routing)
- `layer_replication` (layer expansion)
- `runtime_config` (ephemeral GPU offload)
- Advanced initialization: `eva`, `olora`, `pissa`, `corda`, `loftq`
- `alora_invocation_tokens` (Activated LoRA)

---

## 6. Configuration Sub-Configs Missing

### LoRA Sub-Configs:
- `LoftQConfig`
- `EvaConfig`
- `CordaConfig`
- `ArrowConfig`
- `BdLoraConfig`
- `LoraRuntimeConfig`

---

## 7. Test Coverage Gaps

### Python PEFT Testing:
- 45 test files
- 1500+ test methods
- GPU-specific tests
- Regression tests
- Integration tests

### Elixir Port Testing:
- 9 test files
- 149 tests
- LoRA-focused only

### Testing Gaps:
- No multi-adapter tests
- No merge strategy tests
- No integration tests with Axon
- No state dict save/load tests (proper format)

---

## Implementation Priority Order

### Phase 1: Core Infrastructure (Immediate)
1. `utils/constants.py` - Target module mappings
2. `mapping.py` - Type registry
3. `utils/save_and_load.py` - State dict operations
4. `utils/other.py` - Core utilities
5. `tuners/tuners_utils.py` - Base tuner enhancement

### Phase 2: High-Value Tuners
6. IA3 (simplest after LoRA)
7. Prompt Tuning
8. Prefix Tuning
9. AdaLoRA
10. OFT/BOFT

### Phase 3: LyCORIS Family
11. LoHa
12. LoKr

### Phase 4: Advanced Methods
13. VeRA
14. Poly
15. XLoRA
16. VBLoRA

### Phase 5: Specialized Methods
17-31. Remaining 15 methods

### Phase 6: Advanced Features
32. Mixed model support
33. Auto model loading
34. Merge utilities
35. Hotswap support

---

## File Structure for Implementation Docs

```
docs/20251231/peft_gap_analysis/
├── GAP_ANALYSIS_SUMMARY.md (this file)
├── core/
│   ├── 01_mapping.md
│   ├── 02_tuners_utils.md
│   ├── 03_save_and_load.md
│   └── 04_constants.md
├── tuners/
│   ├── 01_ia3.md
│   ├── 02_prompt_tuning.md
│   ├── 03_prefix_tuning.md
│   ├── 04_adalora.md
│   ├── 05_oft.md
│   ├── 06_boft.md
│   ├── 07_loha.md
│   ├── 08_lokr.md
│   └── ... (remaining tuners)
├── utilities/
│   ├── 01_merge_utils.md
│   ├── 02_other.md
│   └── 03_integrations.md
└── prompts/
    ├── 01_ia3_prompt.md
    ├── 02_prompt_tuning_prompt.md
    └── ... (TDD prompts for each feature)
```

---

## Estimated Effort

| Category | Modules | Estimated Lines | Complexity |
|----------|---------|-----------------|------------|
| Core Infrastructure | 5 | ~2000 | Medium |
| Tuners (30 methods) | 90+ | ~15000 | High |
| Utilities | 11 | ~5000 | Medium |
| Task Models | 6 | ~2000 | Medium |
| Tests | Many | ~5000 | Medium |
| **Total** | **112+** | **~29000** | - |

---

## Next Steps

1. Create detailed implementation docs for each gap item
2. Create TDD prompt files with specific instructions
3. Implement in priority order
4. Maintain test-first development approach
5. Update README.md with each completed feature
