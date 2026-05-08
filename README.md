# NeuralAmp C++

NeuralAmp C++ is an experimental neural amplifier profiling tool. It generates
synthetic amp-capture data, trains a recurrent neural network to imitate a
virtual tube amplifier, and writes checkpoint/model artifacts from a command-line
workflow.

The project is written in C++20 and implements the training stack directly. It
does not use LibTorch, TensorFlow, ONNX Runtime, MATLAB, or any other machine
learning framework.

The runtime now includes a reusable streaming inference state for low-latency
single-sample prediction. This is the path to use when embedding the model in a
larger realtime system.

## Core Idea

The model learns a conditioned audio mapping:

```text
[audio, band_01, band_02, ..., band_20] -> amplifier output
```

The first channel is the input audio. The next 20 channels are sweep controls
for logarithmically spaced frequency bands from 40 Hz to 20 kHz. During dataset
generation, the sweep emphasizes adjacent bands over time with a small
deterministic dither, giving the network explicit coverage across the whole
frequency range.

The virtual target amp applies:

- asymmetric tanh preamp saturation;
- DC blocking;
- a 20-band parallel tone/sweep filter bank;
- power-amp soft clipping;
- cabinet low-pass filtering.

The neural model is:

```text
Sequence input -> GRU -> GRU -> Dense -> ELU -> Dense output
```

## Build Requirements

- CMake 3.24 or newer.
- MSVC 2022 Build Tools on Windows. MSVC 2019 also works with the current code.
- Internet access on first configure so CMake can fetch:
  - Eigen 3.4 for matrix math;
  - nlohmann/json 3.11 for config metadata.

Build and test from the repository root:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

If CMake selects another Visual Studio generator automatically, use the generated
path for the executable.

## Run The Pipeline

Run the default quick pipeline:

```powershell
.\build\Release\neural_amp.exe run --preset quick
```

Run a short smoke test:

```powershell
.\build\Release\neural_amp.exe run --preset quick --duration 1 --epochs 1 --bands 20
```

Generate a dataset without training:

```powershell
.\build\Release\neural_amp.exe generate --preset quick --output experiments\quick_data --bands 20
```

Train from an existing dataset:

```powershell
.\build\Release\neural_amp.exe train --dataset experiments\quick_data --output experiments\quick_train --preset quick
```

Validate a saved model:

```powershell
.\build\Release\neural_amp.exe validate --model experiments\quick_train\final_model.bin --dataset experiments\quick_data
```

Benchmark streaming inference latency:

```powershell
.\build\Release\neural_amp.exe benchmark --model experiments\quick_train\final_model.bin --samples 10000 --warmup 1000
```

Inspect a checkpoint:

```powershell
.\build\Release\neural_amp.exe inspect --checkpoint experiments\quick_train\checkpoints\checkpoint_epoch_0001.bin
```

## Presets

| Preset | Sample rate | Duration | Input | Model | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `quick` | 48 kHz | 2 seconds | audio + 20 bands | GRU 8/4, dense 8 | Fast local verification |
| `gen4-full` | 192 kHz | 180 seconds | audio + 20 bands | GRU 128/64, dense 32 | Full research-style run |

`gen4-full` is CPU-only. It preserves the high-resolution research target, but
it can be very slow compared with framework-backed GPU training.

## Outputs

Runs write to `experiments/<session>/` unless `--output` is supplied.

- `config.json`: resolved preset and CLI settings.
- `dataset.bin`: chunked binary training data.
- `dataset.json`: dataset metadata including band count and input channels.
- `checkpoints/checkpoint_epoch_XXXX.bin`: model weights plus Adam state.
- `final_model.bin`: inference-ready model weights and normalization stats.

Validation is console-only and prints ESR plus an accuracy percentage.

The benchmark command prints mean, p50, p95, p99, max latency, throughput, and a
checksum. It uses the allocation-aware streaming inference path with persistent
GRU state.

## Project Layout

- `src/Dataset.*`: chirp/noise generation, 20-band sweep controls, chunking.
- `src/VirtualTubeAmp.*`: virtual amp target and 20-band filter bank.
- `src/Model.*`: GRU, dense layers, forward pass, backpropagation.
- `src/Benchmark.*`: streaming inference latency measurement.
- `src/Trainer.*`: normalization, truncated BPTT, Adam, checkpointing.
- `src/Pipeline.*`: orchestration for generate, train, run, and validate.
- `src/Artifacts.*`: binary model/checkpoint serialization.
- `tests/test_core.cpp`: CTest unit coverage and gradient sanity checks.

## Current Scope

This is an offline research/training tool with a low-latency inference path. It
is not a trading strategy, execution engine, risk system, or financial advice.
If this technology is later adapted for market data, keep the model interface
separate from order routing, risk limits, audit logging, simulation, and live
execution controls. Latency should be measured in the target deployment process,
on the target hardware, with production-like input rates.

## License

MIT License. See `LICENSE`.
