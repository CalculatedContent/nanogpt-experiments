# Device and Colab Support

Use `--device auto` by default. Resolution order is explicit override, XLA after actual XLA-device initialization, CUDA, MPS, then CPU. MPS uses float32; CUDA uses bf16 only when supported; CPU uses float32. Colab workflows should run `wwgpt device-preflight --device auto` and must not fabricate datasets.
