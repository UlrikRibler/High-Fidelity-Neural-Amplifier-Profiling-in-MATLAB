# Stock Signal DSP

Stock Signal DSP is an experimental C++20 signal-analysis tool for market
candles. It converts chart candles from any interval into a high-sample-rate
soundwave, runs the wave through a neural/DSP analysis pipeline, and writes WAV,
dataset, checkpoint, and model artifacts from a command-line workflow.

The project implements the training stack directly. It does not use LibTorch,
TensorFlow, ONNX Runtime, MATLAB, or another machine learning framework.

The runtime includes a reusable streaming inference state for low-latency
single-sample prediction. This is the path to use when embedding the model in a
larger realtime chart or tool-menu workflow.

## Core Idea

The model learns a conditioned market-signal mapping:

```text
[candle_waveform, band_01, band_02, ..., band_20] -> market signal response
```

The first channel is the candle-derived waveform. The next 20 channels are
frequency-band controls from 0 Hz to 20 kHz. The 0 Hz band captures slow
baseline/trend movement, while the remaining bands are logarithmically spaced
through the audible spectrum. This lets low-timeframe and high-timeframe chart
candles be converted into dense audio-rate samples before analysis.

The candle tool maps OHLCV candles into audio by combining:

- close-to-close return direction;
- candle body and wick/range motion;
- volume/activity pulses;
- volatility and momentum band controls.

The stock signal processor applies:

- a DC/0 Hz low band for trend/baseline information;
- a 20-band market activity filter bank up to 20 kHz;
- dynamic band emphasis from range, momentum, and volume;
- full-band normalization without the previous 40 Hz high-pass behavior.

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

## Candle Sound Tool

Convert chart candles to a high-sample-rate soundwave and dataset:

```powershell
.\build\Release\stock_signal.exe candles --candles path\to\candles.csv --interval 1D --preset candle-analysis --output experiments\candle_1d
```

Supported candle inputs:

- CSV with headers such as `time,open,high,low,close,volume`.
- JSON as an array, `{ "candles": [...] }`, or `{ "series": [...] }`.
- Trading chart-style keys: `open/high/low/close/volume`, `o/h/l/c/v`, or `price_usd`.

Useful options:

```powershell
--sample-rate 192000
--seconds-per-candle 0.02
--chunk-seconds 0.25
--hop-seconds 0.125
--bands 20
--min-hz 0
--max-hz 20000
```

Outputs written to the selected output directory:

- `candle_waveform.wav`: the direct OHLCV-to-sound conversion.
- `market_signal_response.wav`: the processed stock-analysis response.
- `candle_signal.json`: conversion metadata.
- `dataset.bin` and `dataset.json`: neural training dataset.
- `config.json`: resolved preset and CLI settings.

## Run The Pipeline

Generate synthetic market-signal data and train:

```powershell
.\build\Release\stock_signal.exe run --preset quick
```

Run a short smoke test:

```powershell
.\build\Release\stock_signal.exe run --preset quick --duration 1 --epochs 1 --bands 20
```

Generate a dataset without training:

```powershell
.\build\Release\stock_signal.exe generate --preset quick --output experiments\quick_data --bands 20
```

Train from an existing dataset:

```powershell
.\build\Release\stock_signal.exe train --dataset experiments\quick_data --output experiments\quick_train --preset quick
```

Validate a saved model:

```powershell
.\build\Release\stock_signal.exe validate --model experiments\quick_train\final_model.bin --dataset experiments\quick_data
```

Benchmark streaming inference latency:

```powershell
.\build\Release\stock_signal.exe benchmark --model experiments\quick_train\final_model.bin --samples 10000 --warmup 1000
```

Inspect a checkpoint:

```powershell
.\build\Release\stock_signal.exe inspect --checkpoint experiments\quick_train\checkpoints\checkpoint_epoch_0001.bin
```

## Presets

| Preset | Sample rate | Duration | Input | Model | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `quick` | 96 kHz | 2 seconds | waveform + 20 bands | GRU 8/4, dense 8 | Fast local verification |
| `candle-analysis` | 192 kHz | 30 seconds | candle waveform + 20 bands | GRU 64/32, dense 24 | Chart candle sound analysis |
| `gen4-full` | 192 kHz | 180 seconds | waveform + 20 bands | GRU 128/64, dense 32 | Full research-style run |

`gen4-full` is CPU-only. It preserves the high-resolution research target, but
it can be slow compared with framework-backed GPU training.

## Project Layout

- `src/CandleSignal.*`: candle CSV/JSON loading, OHLCV-to-waveform conversion, WAV export.
- `src/Dataset.*`: synthetic market waveform generation, band controls, chunking.
- `src/MarketSignalProcessor.*`: 0-20 kHz stock-analysis filter bank.
- `src/Model.*`: GRU, dense layers, forward pass, backpropagation.
- `src/Benchmark.*`: streaming inference latency measurement.
- `src/Trainer.*`: normalization, truncated BPTT, Adam, checkpointing.
- `src/Pipeline.*`: orchestration for generate, candle conversion, train, run, and validate.
- `src/Artifacts.*`: binary model/checkpoint serialization.
- `tests/test_core.cpp`: CTest unit coverage, candle conversion, and gradient sanity checks.

## Current Scope

This is an offline research/training tool for chart-derived signal analysis. It
is not a trading strategy, execution engine, risk system, or financial advice.
Keep this analysis layer separate from order routing, risk limits, audit
logging, simulation, and live execution controls.

## License

MIT License. See `LICENSE`.
