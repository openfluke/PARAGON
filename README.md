# Paragon

**Parallel Architecture for Resilient Adaptive Growth & Optimized Networks**

Paragon is a high-performance neural network framework written in Go, designed for building resilient and adaptive AI models. It supports dynamic network modification, novel training techniques like replay mechanisms, and a unique Accuracy Deviation Heatmap Distribution (ADHD) metric for performance evaluation. With extensive numerical type support and fixed-point arithmetic, Paragon is ideal for both research and deployment on resource-constrained devices, such as edge hardware.

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
| **Core Neural Network Features**          |             |                |             |          |           |            |
| Dense Layers                              | ✅          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Convolutional Layers                      | 🟡⁴         | ✅             | ✅          | ✅       | ✅        | ✅         |
| Pooling Layers                            | ❌          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Recurrent Layers                          | ❌          | ✅             | ✅          | ✅       | ✅        | ✅         |
| Attention Mechanisms                      | ❌          | ✅             | ✅          | ✅       | ✅        | 🟡⁵        |
| Custom Layer Support                      | ✅⁶         | ✅             | ✅          | ✅       | ✅        | 🟡⁷        |
| Dynamic Network Modification              | ✅⁸         | 🟡⁹            | 🟡⁹         | ❌       | 🟡⁹       | ❌         |
| **Training and Optimization**             |             |                |             |          |           |            |
| Gradient-Based Backpropagation            | ✅          | ✅             | ✅          | 🟡¹⁰     | ✅        | ❌¹¹       |
| Alternative Training Methods              | ✅¹²        | 🟡¹³           | 🟡¹³        | ❌       | 🟡¹³      | ❌         |
| Neural Architecture Search (NAS)          | ✅¹⁴        | 🟡¹⁵           | 🟡¹⁵        | ❌       | 🟡¹⁵      | ❌         |
| Advanced Optimizers (e.g., Adam, RMSprop) | ❌          | ✅             | ✅          | 🟡¹⁰     | ✅        | ❌¹¹       |
| Mixed-Precision Training                  | 🟡¹⁶        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Replay Mechanisms                         | ✅¹⁷        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Early Stopping on Negative Loss           | ✅¹⁸        | ✅             | ✅          | 🟡¹⁰     | ✅        | ❌¹¹       |
| **Performance Evaluation**                |             |                |             |          |           |            |
| ADHD Metric                               | ✅¹⁹        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Detailed Diagnostics                      | ✅²⁰        | ✅             | ✅          | 🟡²¹     | ✅        | 🟡²²       |
| Checkpoint Evaluation                     | ✅²³        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Benchmarking Across Types                 | ✅²⁴        | 🟡²⁵           | 🟡²⁵        | 🟡²⁵     | 🟡²⁵      | 🟡²⁵       |
| **Data Preprocessing**                    |             |                |             |          |           |            |
| Dataset Splitting                         | ✅²⁶        | ✅             | ✅          | ❌       | ✅        | ❌         |
| Data Cleaning/Conversion                  | ✅²⁷        | ✅             | ✅          | ❌       | ✅        | ❌         |
| MNIST Dataset Loading                     | ✅²⁸        | ✅             | ✅          | ✅       | ✅        | ✅         |
| **Persistence and Deployment**            |             |                |             |          |           |            |
| JSON-Based Model Persistence              | ✅²⁹        | 🟡³⁰           | 🟡³⁰        | ✅³¹     | 🟡³⁰      | ❌         |
| Checkpointing                             | ✅³²        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Edge Device Deployment                    | ✅³³        | ✅             | ✅          | ✅       | ✅        | ✅         |
| **Advanced Features**                     |             |                |             |          |           |            |
| Tagged Propagation                        | ✅³⁴        | ❌             | ❌          | ❌       | ❌        | ❌         |
| Sub-Network Support                       | 🟡³⁵        | ✅             | ✅          | ✅       | ✅        | ✅         |
| Concurrent Benchmarking                   | ✅³⁶        | ✅             | ✅          | 🟡³⁷     | ✅        | 🟡³⁷       |

### Notes

1. **TensorFlow**: `int` is supported for indexing but not for core tensor operations.
2. **Paragon**: Supports fixed-point arithmetic with dynamic scaling for integer types, enabling deployment on low-resource devices.
3. **TensorFlow/PyTorch/ONNX/Keras/CoreML**: Fixed-point arithmetic is limited to quantized models (e.g., `int8`), less flexible than Paragon’s generic scaling.
4. **Paragon**: Offers partial convolutional layer support through local receptive fields (`getLocalConnections`), but lacks weight sharing and explicit pooling, resembling a locally connected layer.
5. **CoreML**: Limited support for attention mechanisms, primarily for specific model types.
6. **Paragon**: Custom layers can be implemented via dynamic layer addition and neuron configuration.
7. **CoreML**: Custom layers require significant setup.
8. **Paragon**: Supports dynamic addition of layers and neurons, enabling adaptive architectures.
9. **TensorFlow/PyTorch/Keras**: Dynamic modification is possible but less seamless than Paragon.
10. **ONNX**: Training support depends on the runtime; backpropagation and optimizers are not native.
11. **CoreML**: Focused on inference, not training.
12. **Paragon**: Includes novel training methods like proxy error propagation and bidirectional constraints.
13. **TensorFlow/PyTorch/Keras**: Alternative training methods require custom loops, not built-in like Paragon.
14. **Paragon**: Lightweight Neural Architecture Search (NAS) with activation mutation and weight perturbation.
15. **TensorFlow/PyTorch/Keras**: NAS supported via external libraries (e.g., AutoKeras), not core.
16. **Paragon**: Implicit mixed-precision via generic type support, not optimized for GPUs like other frameworks.
17. **Paragon**: Unique replay mechanisms (manual and dynamic) for repeated layer processing during training.
18. **Paragon**: Early stopping on negative loss detection for training stability.
19. **Paragon**: Accuracy Deviation Heatmap Distribution (ADHD) metric for granular performance analysis.
20. **Paragon**: Detailed diagnostics with metrics like mean absolute error and worst-case errors.
21. **ONNX**: Diagnostics are runtime-dependent, less comprehensive than Paragon.
22. **CoreML**: Diagnostics limited to inference performance metrics.
23. **Paragon**: Supports evaluation from saved checkpoints for iterative training.
24. **Paragon**: Benchmarks across all supported numerical types, as demonstrated with MNIST.
25. **TensorFlow/PyTorch/ONNX/Keras/CoreML**: Benchmarking possible but not as integrated for all numerical types.
26. **Paragon**: Dataset splitting for train/validation sets.
27. **Paragon**: Data cleaning and conversion utilities for preprocessing.
28. **Paragon**: Built-in support for loading and processing MNIST dataset.
29. **Paragon**: JSON-based serialization for model persistence.
30. **TensorFlow/PyTorch/Keras**: Use proprietary formats (SavedModel, PTH); JSON support is limited.
31. **ONNX**: Native support for model interchange, including JSON-like serialization.
32. **Paragon**: Checkpointing for saving and loading layer states.
33. **Paragon**: Fixed-point arithmetic enables efficient edge device deployment.
34. **Paragon**: Tagged propagation for selective computation in large networks.
35. **Paragon**: Sub-network support is limited to single-input cases.
36. **Paragon**: Concurrent benchmarking using Go’s goroutines for efficient type testing.
37. **ONNX/CoreML**: Concurrent benchmarking is runtime-dependent, less integrated.

## Getting Started

The Paragon neural network, part of the **NeuralArena** project, is a flexible and generic neural network implementation in Go, designed to support various numeric types (e.g., `float32`, `float64`, `int8`, `uint8`, etc.) for experimentation with fixed-point and floating-point arithmetic. This guide will walk you through setting up and running the Paragon neural network using the code from the `typeDyn1` branch of the NeuralArena repository, focusing on the MNIST dataset for benchmarking different numeric types and model configurations.

The code is available at:  
[**https://github.com/OpenFluke/NeuralArena/tree/main/typeDyn1**](https://github.com/OpenFluke/NeuralArena/tree/main/typeDyn1)

```go
package main

import (
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

    // Train
    nn.Train(trainInputs, trainTargets, 5, 0.01, true, 5, -5)

    // Evaluate
    var expected, predicted []float64
    for i := range testInputs {
        nn.Forward(testInputs[i])
        nn.ApplySoftmax()
        out := nn.ExtractOutput()
        predicted = append(predicted, float64(paragon.ArgMax(out)))
        expected = append(expected, float64(paragon.ArgMax(testTargets[i][0])))
    }
    nn.EvaluateModel(expected, predicted)
    fmt.Printf("ADHD Score: %.2f\n", nn.Performance.Score)
}
```

See `mnist.go` and `engine.go` in the repository for a full example, including data downloading and concurrent benchmarking.

### Benchmark Results

The following table shows the performance of the Paragon neural network across different numeric types and model configurations, as observed in a recent run:

```
📊 Final Benchmark Results:
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

## Performance Highlights

Paragon’s versatility is demonstrated in MNIST benchmarks:

- **Floating-point types** (`float32`, `float64`) achieve high ADHD scores (~95-96) but take longer (~14-23s).
- **Integer types** (`int`, `uint`, etc.) are faster (~0.7-1.3s) with scores ranging from ~16-44, ideal for edge devices.
- **Replay mechanisms** enhance training flexibility but require tuning for optimal performance.

## Future Goals

Paragon aims to scale and enhance its capabilities. Planned features include:

- **Near-term**: Support for weight sharing and pooling layers to complete convolutional neural network (CNN) functionality, advanced optimizers (e.g., Adam), and improved fixed-point precision for integer types.
- **Long-term**: GPU and NPU acceleration, distributed training across multiple machines, expanded neuron types, multithreading optimizations, WebAssembly for browser execution, and resilience mechanisms for fault tolerance.
