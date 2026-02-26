# vLLM Performance Tuning - Complete Analysis & Recommendations

**Date:** February 26, 2026  
**Platform:** SAP AI Core on T4 GPU (15GB VRAM, SM 7.5)  
**Benchmark Duration:** 2 days of testing across multiple configurations

---

## Executive Summary

Through extensive benchmarking, we identified the optimal vLLM configuration for T4 GPUs:

| Metric | Optimal Config | Worst Config | Improvement |
|--------|---------------|--------------|-------------|
| **Single-User TPS** | **42 TPS** | 4.4 TPS | **10x faster** |
| **Time-to-First-Token** | **133ms** | 516ms | **71% faster** |
| **16-User Aggregate** | **476 TPS** | 41 TPS | **12x faster** |

---

## 1. LLM Engine Choices

### Engine Images Tested

| Engine Image | Version | Result | Status |
|--------------|---------|--------|--------|
| ✅ `vllm/vllm-openai:v0.14.1` | Community | **42 TPS** | **RECOMMENDED** |
| ❌ `nvcr.io/nvidia/nim/vllm:0.12.0` | NVIDIA NGC | 4.4 TPS | 10x slower |

### Why Community vLLM 0.14.1 is Better
- Latest MarlinLinearKernel optimizations
- V1 Engine with better async scheduling
- Full CUDA Graph support
- Better attention backend selection

### Why NVIDIA NGC 0.12.0 is Slower
- Older version missing 2 months of optimizations
- AWQ kernel falls back to slow Exllama
- XFORMERS attention not accepted
- TORCH_SDPA not registered

---

## 2. Model Choices (Full HuggingFace Repo Names)

### ✅ GOOD Models

| Full Model Name (HuggingFace) | Architecture | Single TPS | 16-User TPS | Best For |
|-------------------------------|--------------|------------|-------------|----------|
| `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` | Transformer | **42 TPS** | **476 TPS** | Throughput |
| `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` | Mamba-2 Hybrid | 29 TPS | 50 TPS* | State tracking |
| `casperhansen/llama-3-8b-instruct-awq` | Transformer | 30 TPS | 346 TPS | General purpose |

*Mamba limited by max_num_seqs=16 due to T4 memory constraints

### ❌ BAD Models (on T4 with NGC vLLM 0.12.0)

| Full Model Name (HuggingFace) | Issue | TPS |
|-------------------------------|-------|-----|
| `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | Falls back to slow Exllama kernel on NGC | 4.4 TPS |

### Model Format Comparison

| Quantization Format | Kernel Used | T4 Performance |
|--------------------|-------------|----------------|
| `compressed-tensors` | MarlinLinearKernel | ✅ **Best (42 TPS)** |
| `awq` (Community vLLM) | MarlinLinearKernel | ✅ Good (30 TPS) |
| `awq` (NGC 0.12) | Exllama fallback | ❌ Poor (4.4 TPS) |

---

## 3. Attention Backend Choices

### Backend Compatibility Matrix

| Backend | `vllm/vllm-openai:v0.14.1` | `nvcr.io/nvidia/nim/vllm:0.12.0` | T4 GPU |
|---------|---------------------------|----------------------------------|--------|
| **FLASHINFER** | ✅ Auto-selected | ✅ Auto-selected | ✅ **Best** |
| **XFORMERS** | ✅ Works | ❌ **NOT ACCEPTED** | ⚠️ |
| **TRITON_ATTN** | ✅ Works | ✅ Works | ✅ Good |
| **FLEX_ATTENTION** | ✅ Works | ⚠️ May not exist | ✅ Good |
| **TORCH_SDPA** | ✅ Works | ❌ **NOT REGISTERED** | ⚠️ |
| **FLASH_ATTN (FA2)** | ❌ SM 8.0+ required | ❌ SM 8.0+ required | ❌ **FAILS** |

### Our Experience with Attention Backends

| Attempt | Engine Image | Backend Set | Result |
|---------|--------------|-------------|--------|
| 1 | `nvcr.io/nvidia/nim/vllm:0.12.0` | XFORMERS | ❌ **NOT ACCEPTED** by NVIDIA inference |
| 2 | `nvcr.io/nvidia/nim/vllm:0.12.0` | TORCH_SDPA | ❌ "Backend not registered" error |
| 3 | `nvcr.io/nvidia/nim/vllm:0.12.0` | FLASH_ATTN | ❌ "compute capability >= 8.0" error |
| 4 | `nvcr.io/nvidia/nim/vllm:0.12.0` | (auto) | ✅ FLASHINFER but 4.4 TPS (slow kernel) |
| 5 | `vllm/vllm-openai:v0.14.1` | (auto) | ✅ **FLASHINFER + 42 TPS** |

### ✅ GOOD Attention Choices
- **Let vLLM auto-detect** → Always picks FLASHINFER on T4
- **FLASHINFER** → Works on T4 (SM 7.5), best performance
- **TRITON_ATTN** → Good fallback option

### ❌ BAD Attention Choices
- **XFORMERS on NGC** → NOT ACCEPTED
- **TORCH_SDPA on NGC** → NOT REGISTERED  
- **FLASH_ATTN on T4** → Requires SM 8.0+ (A100/H100 only)
- **Setting `VLLM_ATTENTION_BACKEND` env var** → May force incompatible backend

**Key Takeaway:** Never manually set attention backend on T4. Let vLLM auto-select FLASHINFER.

---

## 4. CUDA Graphs - 71% TTFT Improvement 🔥

### What Are CUDA Graphs?
CUDA Graphs pre-capture GPU execution sequences and replay them without CPU overhead:
- Eliminates kernel launch latency
- Removes CPU-GPU synchronization overhead
- Pre-allocates memory patterns
- Optimizes dynamic scheduling

### Configuration

| Parameter | CUDA Graphs Enabled | CUDA Graphs Disabled |
|-----------|--------------------|-----------------------|
| **enforce_eager** | `False` (default) | `True` |
| **cudagraph_mode** | `FULL_AND_PIECEWISE` | N/A |

### Performance Comparison

| Metric | CUDA Graphs ON | CUDA Graphs OFF | Improvement |
|--------|---------------|-----------------|-------------|
| **TTFT (short prompt)** | **149ms** | 516ms | **-71%** 🔥 |
| **TTFT (medium prompt)** | **133ms** | 457ms | **-71%** 🔥 |
| **Single-User TPS** | 41 TPS | 42 TPS | ~same |
| **16-User Aggregate** | 476 TPS | 493 TPS | -3% |

### CUDA Graph Warmup Cost (One-Time)

| Phase | Time |
|-------|------|
| Dynamo bytecode transform | 9.24s |
| Graph compilation (1-2048 range) | 9.52s |
| PIECEWISE capture (35 graphs) | 3.78s |
| FULL decode capture (19 graphs) | 1.33s |
| **Total warmup** | **~24s** |

### When CUDA Graphs Help Most

| Use Case | Benefit Level |
|----------|--------------|
| **Interactive chat** | 🔥 **HUGE** - 71% faster perceived response |
| **Streaming responses** | 🔥 **HUGE** - First token arrives 3x faster |
| **Low-latency APIs** | ✅ Significant - Consistent sub-200ms TTFT |
| **Batch processing** | ⚠️ Moderate - Similar throughput |

**Key Takeaway:** CUDA Graphs provide the biggest user-perceived improvement. The 71% TTFT improvement makes LLMs feel much more responsive.

---

## 5. Transformer vs Mamba-2 Hybrid Architecture

### Architecture Comparison

| Metric | `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` |
|--------|-------------------------------------------------------|--------------------------------------------------------|
| **Architecture** | Transformer | Mamba-2 Hybrid |
| **Parameters** | 8B | 9B |
| **Single-User TPS** | **42 TPS** | 29 TPS |
| **TPS Consistency (σ)** | 0.5 TPS | **0.3 TPS** (better) |
| **4-User Aggregate** | **158 TPS** | 108 TPS |
| **16-User Aggregate** | **476 TPS** | 50 TPS* |
| **Attention Complexity** | O(n²) | **O(n)** linear |
| **State Tracking** | Standard | **Excellent** |
| **Long Dependencies** | Standard | **Better** |

*Mamba limited by max_num_seqs=16 on T4 due to memory

### Mamba-2 Specific Performance

| Test Type | Avg TPS | Purpose |
|-----------|---------|---------|
| State tracking | 28.64 | SSM sequential state memory |
| Long dependency | 27.72 | Long-range information retrieval |
| Multi-turn state | 27.97 | Conversation context tracking |
| Code reasoning | 28.78 | Complex compute-bound task |
| Numerical chain | 28.84 | Step-by-step calculation |

**Remarkable consistency** across all task types (σ=0.3 TPS)

### When to Use Each Architecture

| Use Case | Best Model | Why |
|----------|-----------|-----|
| **High throughput API** | `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` | 42 TPS, scales to 476 TPS |
| **Multi-turn chat** | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` | Better state tracking |
| **Long context** | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` | O(n) complexity |
| **Sequential reasoning** | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` | SSM state memory |
| **Code generation** | Either | Similar performance |
| **High concurrency (>8)** | `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` | No memory constraints |

---

## 6. Performance Tuning Parameters

### ✅ GOOD Parameters (Use These)

| Parameter | Recommended Value | Impact |
|-----------|------------------|--------|
| **enforce_eager** | `False` | **71% faster TTFT** |
| **cudagraph_mode** | `FULL_AND_PIECEWISE` | Best latency |
| **max_num_batched_tokens** | `2048` | Better batching, higher throughput |
| **max_num_seqs** | `128` | Higher concurrent capacity |
| **enable_chunked_prefill** | `True` | Memory efficient for long prompts |
| **enable_prefix_caching** | `True` | 5-10% improvement on repeated prompts |
| **quantization** | `compressed-tensors` | Enables MarlinLinearKernel |
| **gpu_memory_utilization** | `0.85-0.95` | Maximize KV cache |
| **dtype** | `half` | FP16 for T4 (no BF16 support) |

### ❌ BAD Parameters (Avoid These)

| Parameter | Bad Value | Impact |
|-----------|-----------|--------|
| **enforce_eager** | `True` | 71% slower TTFT |
| **VLLM_ATTENTION_BACKEND** | `XFORMERS` | NOT ACCEPTED by NGC |
| **VLLM_ATTENTION_BACKEND** | `TORCH_SDPA` | NOT REGISTERED in NGC |
| **VLLM_ATTENTION_BACKEND** | `FLASH_ATTN` | FAILS on T4 (SM 7.5) |
| **max_num_batched_tokens** | `512` | Limits batching efficiency |
| **max_num_seqs** | `16` | Caps concurrency too low (for Transformer) |
| **gpu_memory_utilization** | `<0.8` | Wastes KV cache capacity |

### Mamba-Specific Parameters (T4 Memory Constrained)

| Parameter | Value | Why |
|-----------|-------|-----|
| **max_num_seqs** | `16` | 9B Mamba needs more VRAM |
| **max_num_batched_tokens** | `512` | Memory limited |
| **mamba_ssm_cache_dtype** | `float32` | Required for SSM state |
| **gpu_memory_utilization** | `0.85` | Leave headroom for SSM cache |

---

## 7. Final Optimal Configurations

### Configuration A: Maximum Throughput (Transformer)
```yaml
Engine Image: vllm/vllm-openai:v0.14.1
Model: ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ

Parameters:
  --model ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ
  --quantization compressed-tensors
  --dtype half
  --max-model-len 2048
  --max-num-batched-tokens 2048
  --max-num-seqs 128
  --enable-chunked-prefill
  --enable-prefix-caching
  --gpu-memory-utilization 0.9
  # NO --enforce-eager (CUDA Graphs enabled by default)

Expected Performance:
  - Single-user: 42 TPS
  - TTFT: 133ms
  - 16-user aggregate: 476 TPS
```

### Configuration B: State Tracking (Mamba-2 Hybrid)
```yaml
Engine Image: vllm/vllm-openai:v0.14.1
Model: RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16

Parameters:
  --model RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16
  --quantization compressed-tensors
  --dtype half
  --max-model-len 2048
  --max-num-batched-tokens 512
  --max-num-seqs 16
  --enable-chunked-prefill
  --gpu-memory-utilization 0.85
  # NO --enforce-eager (CUDA Graphs enabled by default)

Expected Performance:
  - Single-user: 29 TPS
  - Excellent consistency: σ=0.3 TPS
  - Best for 1-4 concurrent users
  - Superior state tracking and long dependencies
```

---

## 8. Decision Tree

```
Starting a vLLM Deployment on T4?
│
├── Engine Selection
│   └── Use vllm/vllm-openai:v0.14.1 (NOT nvcr.io/nvidia/nim/vllm:0.12.0)
│
├── Model Selection
│   ├── Need high throughput? → ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ (42 TPS)
│   ├── Need state tracking? → RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16 (29 TPS)
│   └── General purpose? → casperhansen/llama-3-8b-instruct-awq (30 TPS)
│
├── Attention Backend
│   └── DO NOT SET MANUALLY → Let vLLM auto-select FLASHINFER
│
├── CUDA Graphs
│   ├── Interactive chat? → ENABLE (enforce_eager=False) 
│   └── Batch processing? → Either works
│
├── Quantization Format
│   └── Use compressed-tensors → Enables MarlinLinearKernel
│
└── Batching Parameters
    ├── Transformer model → max_num_seqs=128, batched_tokens=2048
    └── Mamba model → max_num_seqs=16, batched_tokens=512
```

---

## 9. Key Takeaways

| # | Takeaway | Impact |
|---|----------|--------|
| 1 | Use `vllm/vllm-openai:v0.14.1`, NOT `nvcr.io/nvidia/nim/vllm:0.12.0` | 10x TPS improvement |
| 2 | Enable CUDA Graphs (`enforce_eager=False`) | 71% faster TTFT |
| 3 | Never manually set `VLLM_ATTENTION_BACKEND` on T4 | Prevents crashes |
| 4 | Use `compressed-tensors` quantization format | MarlinLinearKernel |
| 5 | `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` for throughput | 42 TPS |
| 6 | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` for state tracking | 29 TPS |
| 7 | `max_num_seqs=128` for Transformer models | Maximize concurrency |
| 8 | `max_num_seqs=16` for Mamba on T4 | Memory constrained |
| 9 | FLASHINFER is best attention backend for T4 | Auto-selected |
| 10 | XFORMERS NOT ACCEPTED by NGC vLLM | Use Community vLLM instead |
| 11 | FLASH_ATTN requires SM 8.0+ (A100/H100) | Not available on T4 |

---

## Appendix: Benchmark Files

| File | Description |
|------|-------------|
| `benchmark_vllm_results_20260224_161951.json` | `casperhansen/llama-3-8b-instruct-awq` benchmark |
| `benchmark_vllm_results_20260224_204055.json` | `ArtusDev/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-AWQ` benchmark |
| `benchmark_nemotron9b_mamba_20260224_234132.json` | `RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16` benchmark |
| `benchmark_vllm_nvidia_results_20260226_004442.json` | NGC vLLM 0.12.0 benchmark |
| `benchmark_vllm_nvidia_results_20260226_112419.json` | CUDA Graphs comparison benchmark |