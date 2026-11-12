# Phase 3 — KV cache probe: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` on RTX 3070 Laptop (8 GB)

| KV type | context | starts? | weights MiB | KV cache MiB | KiB per token | compute MiB | GPU memory used MiB | note |
|---|---|---|---|---|---|---|---|---|
| f16 | 8,192 | ✅ | 4403.49 | 1024.0 | 128.0 | 124.01 | 5717 |  |
| f16 | 12,288 | ✅ | 4403.49 | 1536.0 | 128.0 | 128.01 | 6233 |  |
| f16 | 16,384 | ❌ | 4403.49 | 2048.0 | 128.0 | 132.01 | 6749 | loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM |
| f16 | 24,576 | ❌ | 4403.49 | 3072.0 | 128.0 | 140.01 | 7609 | loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM |
| f16 | 32,768 | ❌ | 4403.49 | 4096.0 | 128.0 | 148.01 | 7617 | loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM |
| q8_0 | 8,192 | ✅ | 4403.49 | 544.0 | 68.0 | 124.09 | 5237 |  |
| q8_0 | 12,288 | ✅ | 4403.49 | 816.0 | 68.0 | 128.09 | 5513 |  |
| q8_0 | 16,384 | ✅ | 4403.49 | 1088.0 | 68.0 | 132.09 | 5789 |  |
| q8_0 | 24,576 | ❌ | 4403.49 | 1632.0 | 68.0 | 160.09 | 6361 | loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM |
| q8_0 | 32,768 | ❌ | 4403.49 | 2176.0 | 68.0 | 200.09 | 6945 | loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM |
