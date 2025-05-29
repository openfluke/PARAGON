# Paragon

**Parallel Architecture for Resilient Adaptive Growth & Optimized Networks**

Paragon is a high-performance neural network framework written in Go, designed for building resilient and adaptive AI models. It supports dynamic network modification, novel training techniques like replay mechanisms, and a unique Accuracy Deviation Heatmap Distribution (ADHD) metric for performance evaluation. With extensive numerical type support, fixed-point arithmetic, and now WebGPU-accelerated forward passing, Paragon is ideal for research and deployment on diverse platforms, including edge devices and GPU-enabled environments.

## Current Implementation

Paragon is actively developed and supports foundational neural network architectures with the following capabilities:

| **Feature**                               | **Paragon** | **TensorFlow** | **PyTorch** | **ONNX** | **Keras** | **CoreML** |
| ----------------------------------------- | ----------- | -------------- | ----------- | -------- | --------- | ---------- |
| **Numerical Type Support**                |             |                |             |          |           |            |
| `int`                                     | ✅          | 🟡¹            | ❌          | ❌       | ❌        | ❌         |
| `int8`                                    | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| `int16`                                   | ✅          | ❌             | ❌          | ✅       | ❌        | ❌         |
| `int32`                                   | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| `int64`                                   | ✅          | ✅             | ✅          | ✅       | ✅        | ❌         |
| `uint`                                    | ✅          | ❌             | ❌          | ❌       | ❌        | ❌         |
| `uint8`                                   | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| `uint16`                                  | ✅          | ❌             | ❌          | ✅       | ❌        | ❌         |
| `uint32`                                  | ✅          | ❌             | ❌          | ❌       | ❌        | ❌         |
| `uint64`                                  | ✅          | ❌             | ❌          | ❌       | ❌        | ❌         |
| `float32`                                 | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| `float64`                                 | ✅          | ✅             | ✅          | ✅       | ✅        | ❌         |
| **Fixed-Point Arithmetic**                | ✅²         | 🟡³            | 🟡³         | 🟡³      | 🟡³       | 🟡³        |
| **Hardware Acceleration**                 |             |                |             |          |           |            |
| WebGPU Forward Pass                       | ✅⁴         | ✅⁵            | ✅⁵         | 🟡⁶      | ✅⁵       | ❌         |
| **Core Neural Network Features**          |             |                |             |          |           |            |
| Dense Layers                              | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Convolutional Layers                      | 🟡⁷         | ✅             | ✅          | ✅       | ✅        | ✅         |
| Pooling Layers                            | ❌          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Recurrent Layers                          | ❌          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Attention Mechanisms                      | ❌          | ✅             | ✅          | ✅       | ✅        | 🟡⁸        |
| Custom Layer Support                      | ✅⁹         | ✅             | ✅          | ✅       | ✅        | 🟡¹⁰       |
| Dynamic Network Modification              | ✅¹¹        | 🟡¹²           | 🟡¹²        | ❌       | 🟡¹²      | ❌         |
| **Training and Optimization**             |             |                |             |          |           |            |
| Gradient-Based Backpropagation            | ✅          | ✅             | ✅          | 🟡¹³     | ✅        | ❌¹⁴       |
| Alternative Training Methods              | ✅¹⁵        | 🟡¹⁶           | 🟡¹⁶        | ❌       | 🟡¹⁶      | ❌         |
| Neural Architecture Search (NAS)          | ✅¹⁷        | 🟡¹⁸           | 🟡¹⁸        | ❌       | 🟡¹⁸      | ❌         |
| Advanced Optimizers (e.g., Adam, RMSprop) | ❌          | ✅             | ✅          | 🟡¹³     | ✅        | ❌¹⁴       |
| Mixed-Precision Training                  | 🟡¹⁹        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Replay Mechanisms                         | ✅²⁰        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Early Stopping on Negative Loss           | ✅²¹        | ✅             | ✅          | 🟡¹³     | ✅        | ❌¹⁴       |
| **Performance Evaluation**                |             |                |             |          |           |            |
| ADHD Metric                               | ✅²²        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Detailed Diagnostics                      | ✅²³        | ✅             | ✅          | 🟡²⁴     | ✅        | 🟡²⁵       |
| Checkpoint Evaluation                     | ✅²⁶        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Benchmarking Across Types                 | ✅²⁷        | 🟡²⁸           | 🟡²⁸        | 🟡²⁸     | 🟡²⁸      | 🟡²⁸       |
| **Data Preprocessing**                    |             |                |             |          |           |            |
| Dataset Splitting                         | ✅²⁹        | ✅             | ✅          | ❌       | ✅        | ❌         |
| Data Cleaning/Conversion                  | ✅³⁰        | ✅             | ✅          | ❌       | ✅        | ❌         |
| MNIST Dataset Loading                     | ✅³¹        | ✅             | ✅          | ✅       | ✅        | ✅         |
| **Persistence and Deployment**            |             |                |             |          |           |            |
| JSON-Based Model Persistence              | ✅³²        | 🟡³³           | 🟡³³        | ✅³⁴     | 🟡³³      | ❌         |
| Checkpointing                             | ✅³⁵        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Edge Device Deployment                    | ✅³⁶        | ✅             | ✅          | ✅       | ✅        | ✅         |
| **Advanced Features**                     |             |                |             |          |           |            |
| Tagged Propagation                        | ✅³⁷        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Sub-Network Support                       | 🟡³⁸        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Concurrent Benchmarking                   | ✅³⁹        | ✅             | ✅          | 🟡⁴⁰     | ✅        | 🟡⁴⁰       |

### Notes

1. **TensorFlow**: `int` is supported for indexing but not for core tensor operations.
2. **Paragon**: Supports fixed-point arithmetic with dynamic scaling for integer types, enabling deployment on low-resource devices.
3. **TensorFlow/PyTorch/ONNX/Keras/CoreML**: Fixed-point arithmetic is limited to quantized models (e.g., `int8`), less flexible than Paragon’s generic scaling.
4. **Paragon**: Implements WebGPU-accelerated forward pass for `float32`, `int32`, and `uint32` types, with optimized compute pipelines and shader generation for layer-specific computations.
5. **TensorFlow/PyTorch/Keras**: Support GPU acceleration via CUDA or other backends, more mature but less portable than WebGPU.
6. **ONNX**: GPU support depends on runtime (e.g., ONNX Runtime with CUDA or DirectML).
7. **Paragon**: Offers partial convolutional layer support through local receptive fields (`getLocalConnections`), but lacks weight sharing and explicit pooling, resembling a locally connected layer.
8. **CoreML**: Limited support for attention mechanisms, primarily for specific model types.
9. **Paragon**: Custom layers can be implemented via dynamic layer addition and neuron configuration.
10. **CoreML**: Custom layers require significant setup.
11. **Paragon**: Supports dynamic addition of layers and neurons, enabling adaptive architectures.
12. **TensorFlow/PyTorch/Keras**: Dynamic modification is possible but less seamless than Paragon.
13. **ONNX**: Training support depends on the runtime; backpropagation and optimizers are not native.
14. **CoreML**: Focused on inference, not training.
15. **Paragon**: Includes novel training methods like proxy error propagation and bidirectional constraints.
16. **TensorFlow/PyTorch/Keras**: Alternative training methods require custom loops, not built-in like Paragon.
17. **Paragon**: Lightweight Neural Architecture Search (NAS) with activation mutation and weight perturbation.
18. **TensorFlow/PyTorch/Keras**: NAS supported via external libraries (e.g., AutoKeras), not core.
19. **Paragon**: Implicit mixed-precision via generic type support, not fully optimized for GPUs like other frameworks.
20. **Paragon**: Unique replay mechanisms (manual and dynamic) for repeated layer processing during training.
21. **Paragon**: Early stopping on negative loss detection for training stability.
22. **Paragon**: Accuracy Deviation Heatmap Distribution (ADHD) metric for granular performance analysis.
23. **Paragon**: Detailed diagnostics with metrics like mean absolute error and worst-case errors.
24. **ONNX**: Diagnostics are runtime-dependent, less comprehensive than Paragon.
25. **CoreML**: Diagnostics limited to inference performance metrics.
26. **Paragon**: Supports evaluation from saved checkpoints for iterative training.
27. **Paragon**: Benchmarks across all supported numerical types, as demonstrated with MNIST, including GPU-accelerated tests.
28. **TensorFlow/PyTorch/ONNX/Keras/CoreML**: Benchmarking possible but not as integrated for all numerical types.
29. **Paragon**: Dataset splitting for train/validation sets.
30. **Paragon**: Data cleaning and conversion utilities for preprocessing.
31. **Paragon**: Built-in support for loading and processing MNIST dataset.
32. **Paragon**: JSON-based serialization for model persistence.
33. **TensorFlow/PyTorch/Keras**: Use proprietary formats (SavedModel, PTH); JSON support is limited.
34. **ONNX**: Native support for model interchange, including JSON-like serialization.
35. **Paragon**: Checkpointing for saving and loading layer states.
36. **Paragon**: Fixed-point arithmetic and WebGPU support enable efficient edge device and GPU-based deployment.
37. **Paragon**: Tagged propagation for selective computation in large networks.
38. **Paragon**: Sub-network support is limited to single-input cases.
39. **Paragon**: Concurrent benchmarking using Go’s goroutines for efficient type testing.
40. **ONNX/CoreML**: Concurrent benchmarking is runtime-dependent, less integrated.

## Getting Started

The Paragon neural network, part of the **NeuralArena** project, is a flexible and generic neural network implementation in Go, designed to support various numeric types (e.g., `float32`, `float64`, `int8`, `uint8`, etc.) for experimentation with fixed-point and floating-point arithmetic, now enhanced with WebGPU-accelerated forward passing. This guide will walk you through setting up and running the Paragon neural network using the code from the `typeDyn1` branch of the NeuralArena repository, focusing on the MNIST dataset for benchmarking different numeric types and model configurations.

The code is available at:  
[**https://github.com/OpenFluke/NeuralArena/tree/main/typeDyn1**](https://github.com/OpenFluke/NeuralArena/tree/main/typeDyn1)

### Example Usage

```go
package main

import (
    "fmt"
    "paragon"
)

func main() {
    // Load MNIST data
    trainInputs, trainTargets, _ := loadMNISTData("mnist_data", true)
    testInputs, testTargets, _ := loadMNISTData("mnist_data", false)

    // Define network: 28x28 input, 16x16 hidden (local connections), 10x1 output
    layers := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
    acts := []string{"leaky_relu", "leaky_relu", "softmax"}
    full := []bool{true, false, true}
    nn := paragon.NewNetwork[float32](layers, acts, full)

    // Enable WebGPU acceleration
    nn.WebGPUNative = true
    err := nn.InitializeOptimizedGPU()
    if err != nil {
        fmt.Printf("Failed to initialize WebGPU: %v\n", err)
        return
    }
    defer nn.CleanupOptimizedGPU()

    // Train
    nn.Train(trainInputs, trainTargets, 5, 0.01, true, 5, -5)

    // Evaluate
    var expected, predicted []float64
    for i := range testInputs {
        nn.Forward(testInputs[i])
        nn.ApplySoftmax()
        out := nn.GetOutput()
        predicted = append(predicted, float64(paragon.ArgMax(out)))
        expected = append(expected, float64(paragon.ArgMax(testTargets[i][0])))
    }
    nn.EvaluateModel(expected, predicted)
    fmt.Printf("ADHD Score: %.2f\n", nn.Performance.Score)
}
```

See `mnist.go` and `engine.go` in the repository for a full example, including data downloading, concurrent benchmarking, and WebGPU setup.

### WebGPU Acceleration

Paragon now supports WebGPU-accelerated forward passing for `float32`, `int32`, and `uint32` types, leveraging the `go-webgpu` library. Key features include:

- **Optimized Compute Pipelines**: Each layer has a dedicated compute pipeline with custom WGSL shaders for activation functions (e.g., `relu`, `leaky_relu`, `tanh`, `softmax`).
- **Buffer Management**: Efficient handling of input, output, weight, bias, and staging buffers for seamless CPU-GPU data transfer.
- **Pipelined Execution**: Supports double-buffered pipelining to reduce CPU-GPU synchronization overhead.
- **Performance Gains**: As shown in MNIST benchmarks, GPU acceleration significantly reduces computation time for supported types, with speedups of 5.01x to 6.64x compared to CPU.

To enable WebGPU, set `WebGPUNative = true` and call `InitializeOptimizedGPU()` before training or inference. Ensure cleanup with `CleanupOptimizedGPU()` to release resources.

## Performance Highlights

Paragon’s versatility is demonstrated in MNIST benchmarks across CPU and GPU computations:

### CPU Benchmark Results

The following table summarizes the performance of the Paragon neural network across different numeric types and model configurations on CPU, as observed in a recent run:

```
📊 CPU Benchmark Results:
Model          Type       Time         ADHD Score
Standard       uint64     694ms        28.77
Standard       int8       716ms        29.65
Standard       uint       811ms        30.44
Standard       int16      800ms        34.18
Standard       uint32     773ms        33.43
Standard       int64      810ms        38.07
Standard       uint8      859ms        28.62
Standard       int32      886ms        43.55
Standard       int        931ms        41.92
Standard       uint16     885ms        41.11
Replay         uint8      1.105s       28.03
Replay         uint16     1.136s       40.79
DynamicReplay  uint16     1.113s       25.49
DynamicReplay  int        1.153s       29.26
DynamicReplay  uint32     1.13s        28.63
Replay         uint64     1.189s       27.65
Replay         uint32     1.196s       26.98
DynamicReplay  int64      1.212s       23.89
Replay         int32      1.206s       28.88
Replay         int16      1.228s       28.99
DynamicReplay  int32      1.232s       23.66
DynamicReplay  uint8      1.237s       30.10
DynamicReplay  int16      1.248s       26.60
DynamicReplay  uint64     1.177s       40.11
Replay         uint       1.252s       33.93
Replay         int64      1.235s       38.12
DynamicReplay  int8       1.231s       16.33
Replay         int8       1.246s       25.95
DynamicReplay  uint       1.303s       25.22
Replay         int        1.334s       31.55
Standard       float64    14.823s      95.97
Standard       float32    14.883s      95.76
Replay         float64    22.036s      95.57
DynamicReplay  float64    22.23s       95.96
Replay         float32    22.969s      95.69
DynamicReplay  float32    23.14s       95.85
```

#### Observations:

- **Floating-Point Models**: `float32` and `float64` achieve high ADHD scores (~95–96), indicating excellent accuracy on MNIST. However, they are significantly slower (14–23 seconds) due to floating-point arithmetic.
- **Integer Models**: Integer types (`int8`, `uint8`, `int16`, etc.) are much faster (694ms–1.334s) but have lower ADHD scores (16.33–43.55). The best integer performance is from `int32` (43.55) and `int` (41.92) in the `Standard` model.
- **Model Variants**: `Standard` models generally outperform `Replay` and `DynamicReplay` for integer types, possibly due to the overhead of replay mechanisms. For floating-point, all models perform similarly.
- **Note**: Integer models underperform compared to floating-point due to precision limitations in fixed-point arithmetic. Improvements to scaling and gradient handling can further boost their performance.

### GPU Benchmark Results

The following table shows the performance of activation functions tested on the MNIST dataset with 10,000 test samples using WebGPU acceleration, comparing CPU and GPU execution times for `float32`, `int32`, and `uint32` types over 1,000 iterations:

The code is available at:  
[**https://github.com/OpenFluke/NeuralArena/tree/main/gpu1**](https://github.com/OpenFluke/NeuralArena/tree/main/gpu1)

| Type    | Activation | CPU Time     | GPU Time     | Speedup | Match |
| ------- | ---------- | ------------ | ------------ | ------- | ----- |
| float32 | linear     | 5.212794406s | 910.226618ms | 5.73x   | ✅    |
| float32 | relu       | 5.534079058s | 836.45741ms  | 6.62x   | ✅    |
| float32 | leaky_relu | 5.361402801s | 820.887026ms | 6.53x   | ✅    |
| float32 | elu        | 5.533879491s | 843.0522ms   | 6.56x   | ✅    |
| float32 | swish      | 5.401623632s | 865.415472ms | 6.24x   | ✅    |
| float32 | gelu       | 5.375120811s | 859.863697ms | 6.25x   | ✅    |
| float32 | tanh       | 5.293378546s | 797.211397ms | 6.64x   | ✅    |
| float32 | softmax    | 5.425525131s | 1.081928745s | 5.01x   | ✅    |
| int32   | linear     | 5.154817162s | 820.523572ms | 6.28x   | ✅    |
| int32   | relu       | 5.180602845s | 815.494818ms | 6.35x   | ✅    |
| int32   | leaky_relu | 5.038878807s | 838.280008ms | 6.01x   | ✅    |
| int32   | elu        | 5.150231543s | 796.708062ms | 6.46x   | ✅    |
| int32   | swish      | 5.124261151s | 961.447786ms | 5.33x   | ✅    |
| int32   | gelu       | 7.825140508s | 1.20540534s  | 6.49x   | ✅    |
| int32   | tanh       | 7.917425032s | 1.253730796s | 6.32x   | ✅    |
| int32   | softmax    | 6.008736428s | 1.170632946s | 5.13x   | ✅    |
| uint32  | linear     | 5.834971077s | 918.112722ms | 6.36x   | ✅    |
| uint32  | relu       | 5.184863425s | 913.515267ms | 5.68x   | ✅    |
| uint32  | leaky_relu | 5.148608208s | 809.517272ms | 6.36x   | ✅    |
| uint32  | elu        | 5.348548594s | 827.624704ms | 6.46x   | ✅    |
| uint32  | swish      | 5.267901806s | 821.308109ms | 6.41x   | ✅    |
| uint32  | gelu       | 5.552626835s | 915.125859ms | 6.07x   | ✅    |
| uint32  | tanh       | 5.342581747s | 807.581801ms | 6.62x   | ✅    |
| uint32  | softmax    | 5.382638925s | 865.591132ms | 6.22x   | ✅    |

#### Observations:

- **GPU Speedup**: WebGPU acceleration provides significant speedups (5.01x to 6.64x) compared to CPU for all tested activation functions and data types, with `tanh` (6.64x) and `relu` (6.62x) showing the highest gains for `float32`.
- **Data Type Performance**: `float32` generally achieves the lowest GPU times (e.g., 797.211ms for `tanh`), while `int32` and `uint32` are slightly slower but still highly efficient (e.g., 807.581ms for `uint32` `tanh`).
- **Accuracy**: All GPU computations match CPU results (`Match=true`), ensuring correctness of the WebGPU implementation.
- **Note**: A warning was observed during testing: `[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.` This may indicate potential optimization opportunities for specific hardware configurations.

## Future Goals

Paragon aims to scale and enhance its capabilities. Planned features include:

- **Near-term**:
  - Full WebGPU support for backpropagation and training.
  - Support for weight sharing and pooling layers to complete convolutional neural network (CNN) functionality.
  - Advanced optimizers (e.g., Adam, RMSprop).
  - Improved fixed-point precision for integer types to boost ADHD scores.
- **Long-term**:
  - Expanded GPU and NPU acceleration for all numerical types.
  - Distributed training across multiple machines.
  - New neuron types and architectures (e.g., recurrent, attention-based).
  - Multithreading optimizations for CPU performance.
  - WebAssembly support for browser-based execution.
  - Resilience mechanisms for fault tolerance in distributed environments.
