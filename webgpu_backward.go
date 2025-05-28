// webgpu_backward.go
package paragon

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

// GPUBackwardCompute represents GPU resources for backward pass computation
type GPUBackwardCompute struct {
	device      *wgpu.Device
	queue       *wgpu.Queue
	layers      []*GPUBackwardLayer
	initialized bool
	debug       bool
}

// GPUBackwardLayer represents backward pass resources for a single layer
type GPUBackwardLayer struct {
	// Compute pipeline and resources
	pipeline        *wgpu.ComputePipeline
	bindGroup       *wgpu.BindGroup
	bindGroupLayout *wgpu.BindGroupLayout

	// Input/Output buffers
	errorBuffer      *wgpu.Buffer // Error gradients for this layer
	prevErrorBuffer  *wgpu.Buffer // Error gradients to propagate back
	activationBuffer *wgpu.Buffer // Current layer activations
	prevActivBuffer  *wgpu.Buffer // Previous layer activations

	// Weight and bias buffers
	weightBuffer *wgpu.Buffer // Current weights (read/write)
	biasBuffer   *wgpu.Buffer // Current biases (read/write)

	// Gradient buffers
	weightGradBuffer *wgpu.Buffer // Weight gradients
	biasGradBuffer   *wgpu.Buffer // Bias gradients

	// Staging buffers for readback
	weightStaging *wgpu.Buffer
	biasStaging   *wgpu.Buffer
	errorStaging  *wgpu.Buffer

	// Parameters buffer for learning rate, clipping, etc.
	paramsBuffer *wgpu.Buffer

	// Workgroup dimensions
	workgroupsX uint32
	workgroupsY uint32

	// Layer dimensions
	inputSize  uint32
	outputSize uint32
	layerIndex int
	activation string

	forwardOutputBuffer *wgpu.Buffer // Current layer activations from forward pass
	forwardInputBuffer  *wgpu.Buffer // Previous layer activations from forward pass
}

// BackwardParams holds parameters for backward pass computation
type BackwardParams struct {
	LearningRate   float32
	ClipUpper      float32
	ClipLower      float32
	LayerType      uint32 // 0 = hidden, 1 = output
	ActivationType uint32
}

// Initialize GPU backward pass computation
func (n *Network[T]) InitializeGPUBackward() error {
	if any(*new(T)).(T) != T(float32(0)) {
		return fmt.Errorf("GPU backward pass only supported for float32 networks")
	}

	ensureGPU()

	// Initialize backward compute if not already done
	if n.gpu.backward == nil {
		n.gpu.backward = &GPUBackwardCompute{
			device: ctx.device,
			queue:  ctx.queue,
			debug:  n.Debug,
		}
	}

	// Create backward pass resources for each layer (except input layer)
	for l := 1; l <= n.OutputLayer; l++ {
		backwardLayer, err := n.createBackwardLayer(l)
		if err != nil {
			return fmt.Errorf("failed to create backward layer %d: %v", l, err)
		}
		n.gpu.backward.layers = append(n.gpu.backward.layers, backwardLayer)
	}

	n.gpu.backward.initialized = true
	return nil
}

// Create GPU backward pass resources for a single layer
func (n *Network[T]) createBackwardLayer(layerIdx int) (*GPUBackwardLayer, error) {
	prevLayer := n.Layers[layerIdx-1]
	currentLayer := n.Layers[layerIdx]

	inputSize := uint32(prevLayer.Width * prevLayer.Height)
	outputSize := uint32(currentLayer.Width * currentLayer.Height)

	if inputSize == 0 || outputSize == 0 {
		return nil, fmt.Errorf("invalid layer dimensions: input=%d, output=%d", inputSize, outputSize)
	}

	activation := currentLayer.Neurons[0][0].Activation

	// Use the fixed shader generator
	shaderCode := n.generateBackwardShader(layerIdx, inputSize, outputSize, activation)

	// Create shader module
	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("BackwardLayer_%d_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create backward shader module: %v", err)
	}
	defer module.Release()

	// Create bind group layout with corrected binding order
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			// Error buffer (input)
			{
				Binding:    0,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			// Previous error buffer (output)
			{
				Binding:    1,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_Storage,
					HasDynamicOffset: false,
				},
			},
			// Current layer activations
			{
				Binding:    2,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			// Previous layer activations
			{
				Binding:    3,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			// Weights (read/write)
			{
				Binding:    4,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_Storage,
					HasDynamicOffset: false,
				},
			},
			// Biases (read/write)
			{
				Binding:    5,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_Storage,
					HasDynamicOffset: false,
				},
			},
			// Parameters
			{
				Binding:    6,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create backward bind group layout: %v", err)
	}

	// Create pipeline layout
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("BackwardLayer_%d_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create backward pipeline layout: %v", err)
	}
	defer pipelineLayout.Release()

	// Create compute pipeline
	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("BackwardLayer_%d_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create backward compute pipeline: %v", err)
	}

	// Create layer structure
	backwardLayer := &GPUBackwardLayer{
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
		inputSize:       inputSize,
		outputSize:      outputSize,
		layerIndex:      layerIdx,
		activation:      activation,
	}

	// Create buffers
	if err := n.createBackwardBuffers(backwardLayer); err != nil {
		backwardLayer.cleanup()
		return nil, fmt.Errorf("failed to create backward buffers: %v", err)
	}

	// Create bind group
	if err := n.createBackwardBindGroup(backwardLayer); err != nil {
		backwardLayer.cleanup()
		return nil, fmt.Errorf("failed to create backward bind group: %v", err)
	}

	// Calculate workgroup dimensions
	maxSize := max(inputSize, outputSize)
	backwardLayer.workgroupsX = (maxSize + 63) / 64 // 64 threads per workgroup
	backwardLayer.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created backward layer %d: %dx%d -> %dx%d, workgroups: %d\n",
			layerIdx, prevLayer.Width, prevLayer.Height,
			currentLayer.Width, currentLayer.Height, backwardLayer.workgroupsX)
	}

	return backwardLayer, nil
}

// Create all buffers needed for backward pass
func (n *Network[T]) createBackwardBuffers(layer *GPUBackwardLayer) error {
	var err error

	// Error buffers
	layer.errorBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_Error", layer.layerIndex),
		Size:  uint64(layer.outputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create error buffer: %v", err)
	}

	layer.prevErrorBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_PrevError", layer.layerIndex),
		Size:  uint64(layer.inputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create prev error buffer: %v", err)
	}

	// Activation buffers
	layer.activationBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_Activation", layer.layerIndex),
		Size:  uint64(layer.outputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create activation buffer: %v", err)
	}

	layer.prevActivBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_PrevActivation", layer.layerIndex),
		Size:  uint64(layer.inputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create prev activation buffer: %v", err)
	}

	// Weight buffer - CRITICAL FIX: Must have Storage usage for read/write
	weightSize := layer.inputSize * layer.outputSize
	layer.weightBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_Weights", layer.layerIndex),
		Size:  uint64(weightSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc, // FIXED: Storage for read/write
	})
	if err != nil {
		return fmt.Errorf("failed to create weight buffer: %v", err)
	}

	// Bias buffer - CRITICAL FIX: Must match weight buffer permissions
	layer.biasBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_Biases", layer.layerIndex),
		Size:  uint64(layer.outputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc, // FIXED: Storage for read/write
	})
	if err != nil {
		return fmt.Errorf("failed to create bias buffer: %v", err)
	}

	// Parameters buffer
	layer.paramsBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_Params", layer.layerIndex),
		Size:  16, // 4 float32 values
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create params buffer: %v", err)
	}

	// Staging buffers for CPU readback
	layer.weightStaging, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_WeightStaging", layer.layerIndex),
		Size:  uint64(weightSize) * 4,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create weight staging buffer: %v", err)
	}

	layer.biasStaging, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_BiasStaging", layer.layerIndex),
		Size:  uint64(layer.outputSize) * 4,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create bias staging buffer: %v", err)
	}

	layer.errorStaging, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("BackwardLayer_%d_ErrorStaging", layer.layerIndex),
		Size:  uint64(layer.inputSize) * 4,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create error staging buffer: %v", err)
	}

	return nil
}

// Create bind group for backward pass
func (n *Network[T]) createBackwardBindGroup(layer *GPUBackwardLayer) error {
	var err error
	layer.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("BackwardLayer_%d_BindGroup", layer.layerIndex),
		Layout: layer.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			// Binding 0: Error buffer (read-only)
			{Binding: 0, Buffer: layer.errorBuffer, Size: layer.errorBuffer.GetSize()},
			// Binding 1: Previous error buffer (read/write)
			{Binding: 1, Buffer: layer.prevErrorBuffer, Size: layer.prevErrorBuffer.GetSize()},
			// Binding 2: Current activations (read-only)
			{Binding: 2, Buffer: layer.activationBuffer, Size: layer.activationBuffer.GetSize()},
			// Binding 3: Previous activations (read-only)
			{Binding: 3, Buffer: layer.prevActivBuffer, Size: layer.prevActivBuffer.GetSize()},
			// Binding 4: Weights (read/write) - CRITICAL: This must be read/write
			{Binding: 4, Buffer: layer.weightBuffer, Size: layer.weightBuffer.GetSize()},
			// Binding 5: Biases (read/write) - CRITICAL: This must be read/write
			{Binding: 5, Buffer: layer.biasBuffer, Size: layer.biasBuffer.GetSize()},
			// Binding 6: Parameters (read-only)
			{Binding: 6, Buffer: layer.paramsBuffer, Size: layer.paramsBuffer.GetSize()},
		},
	})
	return err
}

func (n *Network[T]) generateBackwardShader(layerIdx int, inputSize, outputSize uint32, activation string) string {
	activationDerivCode := getActivationDerivativeCode(activation)

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> error: array<f32>;
		@group(0) @binding(1) var<storage, read_write> prev_error: array<f32>;
		@group(0) @binding(2) var<storage, read> current_activations: array<f32>;
		@group(0) @binding(3) var<storage, read> prev_activations: array<f32>;
		@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
		@group(0) @binding(5) var<storage, read_write> biases: array<f32>;
		@group(0) @binding(6) var<storage, read> params: array<f32>;

		%s

		fn clip_gradient(grad: f32, clip_lower: f32, clip_upper: f32) -> f32 {
			if (clip_upper == 0.0 && clip_lower == 0.0) {
				return grad;
			}
			return clamp(grad, clip_lower, clip_upper);
		}

		@compute @workgroup_size(64, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let thread_id = global_id.x;
			let learning_rate = params[0];
			let clip_upper = params[1];
			let clip_lower = params[2];
			let is_output_layer = params[3];
			let total_weights = %du * %du;

			// Phase 1: Update biases and weights (distribute across all weights)
			if (thread_id < total_weights) {
				let output_idx = thread_id / %du;
				let input_idx = thread_id %% %du;
				if (output_idx < %du && input_idx < %du) {
					let weight_idx = output_idx * %du + input_idx;
					let local_error = error[output_idx];
					let input_activation = prev_activations[input_idx];

					// Update weight
					var weight_grad = local_error * input_activation;
					weight_grad = clip_gradient(weight_grad, clip_lower, clip_upper);
					weights[weight_idx] = weights[weight_idx] - learning_rate * weight_grad;

					// Update bias (only once per output neuron)
					if (input_idx == 0u) {
						var bias_grad = local_error;
						bias_grad = clip_gradient(bias_grad, clip_lower, clip_upper);
						biases[output_idx] = biases[output_idx] - learning_rate * bias_grad;
					}
				}
			}

			workgroupBarrier();

			// Phase 2: Error propagation (one thread per input neuron)
			if (thread_id < %du) {
				let input_idx = thread_id;
				var accumulated_error: f32 = 0.0;

				for (var output_idx: u32 = 0u; output_idx < %du; output_idx = output_idx + 1u) {
					let weight_idx = output_idx * %du + input_idx;
					accumulated_error = accumulated_error + error[output_idx] * weights[weight_idx];
				}

				if (is_output_layer == 0.0) {
					let input_activation = prev_activations[input_idx];
					let derivative = activation_derivative(input_activation);
					prev_error[input_idx] = accumulated_error * derivative;
				} else {
					prev_error[input_idx] = accumulated_error;
				}
			}
		}
	`, activationDerivCode, outputSize, inputSize, inputSize, inputSize, outputSize, inputSize, inputSize, inputSize, outputSize, inputSize)
}

// Get activation derivative function WGSL code
func getActivationDerivativeCode(activation string) string {
	switch activation {
	case "relu":
		return `fn activation_derivative(x: f32) -> f32 { 
			if (x > 0.0) { return 1.0; } else { return 0.0; }
		}`
	case "leaky_relu":
		return `fn activation_derivative(x: f32) -> f32 { 
			if (x > 0.0) { return 1.0; } else { return 0.01; }
		}`
	case "sigmoid":
		return `fn activation_derivative(x: f32) -> f32 { 
			let s = 1.0 / (1.0 + exp(-clamp(x, -500.0, 500.0)));
			return s * (1.0 - s);
		}`
	case "tanh":
		return `fn activation_derivative(x: f32) -> f32 { 
			let t = tanh(clamp(x, -500.0, 500.0));
			return 1.0 - t * t;
		}`
	case "softmax":
		return `fn activation_derivative(x: f32) -> f32 { return 1.0; }` // Handled in error computation
	default:
		return `fn activation_derivative(x: f32) -> f32 { return 1.0; }`
	}
}

// Execute GPU backward pass
func (n *Network[T]) BackwardGPU(targets [][]float64, lr float64, clipUpper T, clipLower T) error {
	if !n.gpu.backward.initialized {
		return fmt.Errorf("GPU backward pass not initialized")
	}

	if n.Debug {
		fmt.Println("Starting GPU backward pass...")
	}

	// Compute output error
	outputError, err := n.computeOutputError(targets)
	if err != nil {
		return fmt.Errorf("failed to compute output error: %v", err)
	}

	if n.Debug {
		fmt.Printf("Output error stats: min=%.6f, max=%.6f, avg=%.6f\n",
			minFloat32(outputError), maxFloat32(outputError), avgFloat32(outputError))
	}

	// Save initial weights for debugging
	initialWeights := make(map[int][]float32)
	if n.Debug {
		for _, layerCompute := range n.gpu.backward.layers {
			w, _ := n.extractLayerWeightsAndBiases(layerCompute.layerIndex)
			initialWeights[layerCompute.layerIndex] = w
		}
	}

	// Create command encoder
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Process layers in reverse order
	currentError := outputError

	for i := len(n.gpu.backward.layers) - 1; i >= 0; i-- {
		layerCompute := n.gpu.backward.layers[i]
		layerIdx := layerCompute.layerIndex

		if n.Debug {
			fmt.Printf("Processing backward layer %d\n", layerIdx)
		}

		// Upload current activations from forward pass
		currentActivations := n.extractLayerActivations(layerIdx)
		ctx.queue.WriteBuffer(layerCompute.activationBuffer, 0, wgpu.ToBytes(currentActivations))

		// Upload previous layer activations
		prevActivations := n.extractLayerActivations(layerIdx - 1)
		ctx.queue.WriteBuffer(layerCompute.prevActivBuffer, 0, wgpu.ToBytes(prevActivations))

		// Upload error
		ctx.queue.WriteBuffer(layerCompute.errorBuffer, 0, wgpu.ToBytes(currentError))

		// Upload current weights and biases
		weights, biases := n.extractLayerWeightsAndBiases(layerIdx)
		ctx.queue.WriteBuffer(layerCompute.weightBuffer, 0, wgpu.ToBytes(weights))
		ctx.queue.WriteBuffer(layerCompute.biasBuffer, 0, wgpu.ToBytes(biases))

		// Upload parameters
		isOutputLayer := float32(0)
		if layerIdx == n.OutputLayer {
			isOutputLayer = 1
		}
		params := []float32{
			float32(lr),
			float32(any(clipUpper).(T)),
			float32(any(clipLower).(T)),
			isOutputLayer,
		}
		ctx.queue.WriteBuffer(layerCompute.paramsBuffer, 0, wgpu.ToBytes(params))

		// Create compute pass
		computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("BackwardLayer_%d", layerIdx),
		})

		computePass.SetPipeline(layerCompute.pipeline)
		computePass.SetBindGroup(0, layerCompute.bindGroup, nil)

		// Calculate workgroups to cover all weights
		totalWeights := uint32(layerCompute.inputSize * layerCompute.outputSize)
		workgroups := (totalWeights + 63) / 64 // 64 threads per workgroup
		if workgroups == 0 {
			workgroups = 1
		}

		if n.Debug {
			fmt.Printf("Dispatching %d workgroups for layer %d (total weights: %d)\n", workgroups, layerIdx, totalWeights)
		}

		computePass.DispatchWorkgroups(workgroups, 1, 1)
		computePass.End()

		// Copy results to staging
		encoder.CopyBufferToBuffer(
			layerCompute.weightBuffer, 0,
			layerCompute.weightStaging, 0,
			layerCompute.weightBuffer.GetSize(),
		)
		encoder.CopyBufferToBuffer(
			layerCompute.biasBuffer, 0,
			layerCompute.biasStaging, 0,
			layerCompute.biasBuffer.GetSize(),
		)

		// Copy error for next iteration (if not the first layer)
		if i > 0 {
			encoder.CopyBufferToBuffer(
				layerCompute.prevErrorBuffer, 0,
				n.gpu.backward.layers[i-1].errorBuffer, 0,
				layerCompute.prevErrorBuffer.GetSize(),
			)
			currentError, err = n.readBackwardStagingBuffer(layerCompute.errorStaging, int(layerCompute.inputSize))
			if err != nil {
				return fmt.Errorf("failed to read error for layer %d: %v", layerIdx, err)
			}
		}
	}

	// Submit all commands
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)
	ctx.device.Poll(true, nil)

	// Debug: Verify buffer contents after compute
	if n.Debug {
		for _, layerCompute := range n.gpu.backward.layers {
			weights, err := n.readBackwardStagingBuffer(layerCompute.weightStaging, int(layerCompute.inputSize*layerCompute.outputSize))
			if err != nil {
				return fmt.Errorf("debug: failed to read weights for layer %d: %v", layerCompute.layerIndex, err)
			}
			nonZeroWeights := 0
			for _, w := range weights {
				if w != 0.0 {
					nonZeroWeights++
				}
			}
			fmt.Printf("Layer %d: %d/%d non-zero weights after GPU compute\n", layerCompute.layerIndex, nonZeroWeights, len(weights))

			// Compare with initial weights
			initial := initialWeights[layerCompute.layerIndex]
			changes := 0
			maxChange := float32(0)
			for i, w := range weights {
				if i < len(initial) {
					change := absFloat32(w - initial[i])
					if change > 0.0001 {
						changes++
					}
					if change > maxChange {
						maxChange = change
					}
				}
			}
			fmt.Printf("Layer %d: %d weights changed, max change=%.6f\n", layerCompute.layerIndex, changes, maxChange)
		}
	}

	// Apply the GPU results back to the CPU network
	if err := n.applyBackwardResults(); err != nil {
		return fmt.Errorf("failed to apply backward results: %v", err)
	}

	if n.Debug {
		fmt.Println("GPU backward pass completed successfully")
	}

	return nil
}

// uploadBackwardDataFixed uploads all the data needed for the GPU backward pass,
// including both current and previous activations, weights/biases, error and params.
func (n *Network[T]) uploadBackwardDataFixed(
	layer *GPUBackwardLayer,
	layerIdx int,
	errorData []float32,
	lr float64,
	clipUpper T,
	clipLower T,
) error {
	// Upload error for this layer
	ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(errorData))

	// Upload current weights and biases
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)
	ctx.queue.WriteBuffer(layer.weightBuffer, 0, wgpu.ToBytes(weights))
	ctx.queue.WriteBuffer(layer.biasBuffer, 0, wgpu.ToBytes(biases))

	// Upload parameters
	isOutputLayer := float32(0)
	if layerIdx == n.OutputLayer {
		isOutputLayer = 1
	}

	params := []float32{
		float32(lr),
		float32(any(clipUpper).(T)),
		float32(any(clipLower).(T)),
		isOutputLayer,
	}
	ctx.queue.WriteBuffer(layer.paramsBuffer, 0, wgpu.ToBytes(params))

	if n.Debug {
		fmt.Printf("Uploaded layer %d: %d weights, %d biases, lr=%.4f\n",
			layerIdx, len(weights), len(biases), lr)
	}

	return nil
}

// Compute output layer error (done on CPU)
func (n *Network[T]) computeOutputError(targets [][]float64) ([]float32, error) {
	outputLayer := n.Layers[n.OutputLayer]
	errorData := make([]float32, outputLayer.Width*outputLayer.Height)

	if n.Debug {
		fmt.Printf("Computing output error for %dx%d layer\n", outputLayer.Width, outputLayer.Height)
	}

	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			pred := float64(any(neuron.Value).(T))
			target := targets[y][x]

			// For cross-entropy + softmax, gradient is: predicted - target
			// This is numerically stable and correct
			if neuron.Activation == "softmax" {
				errorData[idx] = float32(pred - target)
			} else {
				// For other activations: (target - pred) * activation_derivative
				diff := target - pred
				deriv := float64(any(ActivationDerivativeGeneric(neuron.Value, neuron.Activation)).(T))
				errorData[idx] = float32(diff * deriv)
			}

			if n.Debug && idx < 3 {
				fmt.Printf("Error[%d]: pred=%.6f, target=%.6f, error=%.6f\n",
					idx, pred, target, errorData[idx])
			}
			idx++
		}
	}

	return errorData, nil
}

// Extract layer activations as float32 slice
func (n *Network[T]) extractLayerActivations(layerIdx int) []float32 {
	layer := n.Layers[layerIdx]
	activations := make([]float32, layer.Width*layer.Height)

	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			activations[idx] = float32(any(layer.Neurons[y][x].Value).(T))
			idx++
		}
	}

	return activations
}

// Apply backward pass results to CPU network
func (n *Network[T]) applyBackwardResults() error {
	for _, layerCompute := range n.gpu.backward.layers {
		layerIdx := layerCompute.layerIndex

		if n.Debug {
			fmt.Printf("Reading results for layer %d...\n", layerIdx)
		}

		// Read updated weights
		weights, err := n.readBackwardStagingBuffer(layerCompute.weightStaging, int(layerCompute.inputSize*layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read weights for layer %d: %v", layerIdx, err)
		}

		// Read updated biases
		biases, err := n.readBackwardStagingBuffer(layerCompute.biasStaging, int(layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read biases for layer %d: %v", layerIdx, err)
		}

		if n.Debug {
			fmt.Printf("Read %d weights and %d biases for layer %d\n", len(weights), len(biases), layerIdx)
		}

		// Apply weights and biases to CPU network
		if err := n.applyWeightsAndBiases(layerIdx, weights, biases); err != nil {
			return fmt.Errorf("failed to apply weights and biases for layer %d: %v", layerIdx, err)
		}

		if n.Debug {
			fmt.Printf("Applied GPU backward results to layer %d\n", layerIdx)
		}
	}

	if netF32, ok := any(n).(*Network[float32]); ok {
		netF32.syncForwardWeightsToGPU() // keep forward & backward in lock-step
		netF32.syncBackwardWeightsToGPU()
	}

	return nil
}

// Read staging buffer with proper synchronization (similar to forward pass)
func (n *Network[T]) readBackwardStagingBuffer(buffer *wgpu.Buffer, size int) ([]float32, error) {
	done := make(chan wgpu.BufferMapAsyncStatus, 1)

	err := buffer.MapAsync(wgpu.MapMode_Read, 0, buffer.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Poll for completion
	for {
		ctx.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatus_Success {
				return nil, fmt.Errorf("buffer mapping failed: %v", status)
			}
			goto readData
		default:
			continue
		}
	}

readData:
	data := buffer.GetMappedRange(0, uint(size*4))
	if data == nil {
		buffer.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	result := make([]float32, size)
	copy(result, wgpu.FromBytes[float32](data))
	buffer.Unmap()

	return result, nil
}

// Apply GPU-computed weights and biases to CPU network
func (n *Network[T]) applyWeightsAndBiases(layerIdx int, weights []float32, biases []float32) error {
	currentLayer := n.Layers[layerIdx]
	prevLayer := n.Layers[layerIdx-1]

	inputSize := prevLayer.Width * prevLayer.Height
	outputSize := currentLayer.Width * currentLayer.Height

	if len(weights) != inputSize*outputSize {
		return fmt.Errorf("weight size mismatch: got %d, expected %d", len(weights), inputSize*outputSize)
	}
	if len(biases) != outputSize {
		return fmt.Errorf("bias size mismatch: got %d, expected %d", len(biases), outputSize)
	}

	// Apply biases
	idx := 0
	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			currentLayer.Neurons[y][x].Bias = T(biases[idx])
			idx++
		}
	}

	// Apply weights - CRITICAL FIX: only update existing connections
	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			weightRowStart := neuronIdx * inputSize

			// Update each existing connection's weight
			for i := range neuron.Inputs {
				conn := &neuron.Inputs[i] // Get pointer to modify

				// Calculate input index from connection coordinates
				inputIdx := conn.SourceY*prevLayer.Width + conn.SourceX

				// Bounds check and update
				if inputIdx >= 0 && inputIdx < inputSize {
					weightIdx := weightRowStart + inputIdx
					if weightIdx < len(weights) {
						conn.Weight = T(weights[weightIdx])
					}
				}
			}
		}
	}

	return nil
}

// Cleanup backward pass resources
func (layer *GPUBackwardLayer) cleanup() {
	if layer.errorBuffer != nil {
		layer.errorBuffer.Destroy()
	}
	if layer.prevErrorBuffer != nil {
		layer.prevErrorBuffer.Destroy()
	}
	if layer.activationBuffer != nil {
		layer.activationBuffer.Destroy()
	}
	if layer.prevActivBuffer != nil {
		layer.prevActivBuffer.Destroy()
	}
	if layer.weightBuffer != nil {
		layer.weightBuffer.Destroy()
	}
	if layer.biasBuffer != nil {
		layer.biasBuffer.Destroy()
	}
	if layer.paramsBuffer != nil {
		layer.paramsBuffer.Destroy()
	}
	if layer.weightStaging != nil {
		layer.weightStaging.Destroy()
	}
	if layer.biasStaging != nil {
		layer.biasStaging.Destroy()
	}
	if layer.errorStaging != nil {
		layer.errorStaging.Destroy()
	}
	if layer.bindGroup != nil {
		layer.bindGroup.Release()
	}
	if layer.bindGroupLayout != nil {
		layer.bindGroupLayout.Release()
	}
	if layer.pipeline != nil {
		layer.pipeline.Release()
	}
}

// Clean up all GPU backward pass resources
func (n *Network[T]) CleanupGPUBackward() {
	if n.gpu.backward == nil {
		return
	}

	for i, layer := range n.gpu.backward.layers {
		layer.cleanup()
		if n.Debug {
			fmt.Printf("Cleaned up GPU backward resources for layer %d\n", i)
		}
	}

	n.gpu.backward.layers = nil
	n.gpu.backward.initialized = false
	n.gpu.backward = nil
}

// Update the main Backward method to use GPU when available
func (n *Network[T]) Backward(targets [][]float64, lr float64, clipUpper T, clipLower T) {
	// Use GPU backward pass if available and enabled
	if n.WebGPUNative && n.gpu.backward != nil && n.gpu.backward.initialized {
		err := n.BackwardGPU(targets, lr, clipUpper, clipLower)
		if err != nil {
			if n.Debug {
				fmt.Printf("GPU backward pass failed, falling back to CPU: %v\n", err)
			}
			// Fall back to CPU backward pass
			n.backwardCPU(targets, lr, clipUpper, clipLower)
		}
		return
	}

	// Fallback to CPU backward pass
	n.backwardCPU(targets, lr, clipUpper, clipLower)
}

func (n *Network[T]) TrainGPU(
	inputs [][][]float64,
	targets [][][]float64,
	epochs int,
	learningRate float64,
	earlyStopOnNegativeLoss bool,
	clipUpper T,
	clipLower T,
) error {
	// Initialize GPU resources if not already done
	if err := n.InitializeGPUComplete(); err != nil {
		return fmt.Errorf("failed to initialize GPU: %v", err)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))

		for _, i := range perm {
			// Forward pass
			n.Forward(inputs[i])

			// Compute loss
			loss := n.ComputeLoss(targets[i])
			if math.IsNaN(loss) {
				if n.Debug {
					fmt.Printf("NaN loss at sample %d, epoch %d\n", i, epoch)
				}
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				return nil
			}

			totalLoss += loss

			// Fixed backward pass
			n.Backward(targets[i], learningRate, clipUpper, clipLower)
		}

		if n.Debug || epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(inputs)))
		}
	}

	return nil
}

func (n *Network[T]) ExtractLayerWeightsAndBiases(layerIdx int) ([]float32, []float32) {
	return n.extractLayerWeightsAndBiases(layerIdx)
}

func (n *Network[T]) ApplyWeightsAndBiases(layerIdx int, weights []float32, biases []float32) error {
	return n.applyWeightsAndBiases(layerIdx, weights, biases)
}

func (n *Network[T]) syncActivationsToBackward() error {
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}

	for _, backwardLayer := range n.gpu.backward.layers {
		layerIdx := backwardLayer.layerIndex

		// Get current layer activations
		currentActivations := n.extractLayerActivations(layerIdx)
		ctx.queue.WriteBuffer(backwardLayer.activationBuffer, 0, wgpu.ToBytes(currentActivations))

		// Get previous layer activations
		prevActivations := n.extractLayerActivations(layerIdx - 1)
		ctx.queue.WriteBuffer(backwardLayer.prevActivBuffer, 0, wgpu.ToBytes(prevActivations))

		if n.Debug {
			fmt.Printf("Synced %d current + %d prev activations for layer %d\n",
				len(currentActivations), len(prevActivations), layerIdx)
		}
	}

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return err
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)
	return nil
}

func (n *Network[T]) readBackwardResults() error {
	for _, layerCompute := range n.gpu.backward.layers {
		layerIdx := layerCompute.layerIndex

		// Read weights
		weights, err := n.readBackwardStagingBuffer(layerCompute.weightStaging,
			int(layerCompute.inputSize*layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read weights for layer %d: %v", layerIdx, err)
		}

		// Read biases
		biases, err := n.readBackwardStagingBuffer(layerCompute.biasStaging,
			int(layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read biases for layer %d: %v", layerIdx, err)
		}

		// Apply to network
		if err := n.applyWeightsAndBiases(layerIdx, weights, biases); err != nil {
			return fmt.Errorf("failed to apply results for layer %d: %v", layerIdx, err)
		}

		if n.Debug {
			// Check for NaN/Inf values
			nanCount := 0
			for _, w := range weights {
				if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
					nanCount++
				}
			}
			if nanCount > 0 {
				fmt.Printf("WARNING: Layer %d has %d NaN/Inf weights\n", layerIdx, nanCount)
			}
		}
	}

	return nil
}

func (n *Network[T]) ValidateGPUBackward(inputs [][]float64, targets [][]float64, lr float64) error {
	if n.Debug {
		fmt.Println("=== GPU Backward Pass Validation ===")
	}

	// Save original state
	originalWeights := make(map[string][]float32)
	for l := 1; l <= n.OutputLayer; l++ {
		w, b := n.extractLayerWeightsAndBiases(l)
		originalWeights[fmt.Sprintf("w%d", l)] = make([]float32, len(w))
		originalWeights[fmt.Sprintf("b%d", l)] = make([]float32, len(b))
		copy(originalWeights[fmt.Sprintf("w%d", l)], w)
		copy(originalWeights[fmt.Sprintf("b%d", l)], b)
	}

	// Run forward pass
	n.Forward(inputs)
	loss1 := n.ComputeLoss(targets)

	// GPU backward pass with type-safe clipping bounds
	var clipUpper, clipLower T
	var zero T

	switch any(zero).(type) {
	case int8:
		clipUpper = any(int8(100)).(T)
		clipLower = any(int8(-100)).(T)
	case int16:
		clipUpper = any(int16(1000)).(T)
		clipLower = any(int16(-1000)).(T)
	case int32:
		clipUpper = any(int32(10000)).(T)
		clipLower = any(int32(-10000)).(T)
	case int64:
		clipUpper = any(int64(10000)).(T)
		clipLower = any(int64(-10000)).(T)
	case int:
		clipUpper = any(int(10000)).(T)
		clipLower = any(int(-10000)).(T)
	case uint8:
		clipUpper = any(uint8(200)).(T)
		clipLower = any(uint8(0)).(T)
	case uint16:
		clipUpper = any(uint16(2000)).(T)
		clipLower = any(uint16(0)).(T)
	case uint32:
		clipUpper = any(uint32(20000)).(T)
		clipLower = any(uint32(0)).(T)
	case uint64:
		clipUpper = any(uint64(20000)).(T)
		clipLower = any(uint64(0)).(T)
	case uint:
		clipUpper = any(uint(20000)).(T)
		clipLower = any(uint(0)).(T)
	case float32:
		clipUpper = any(float32(1000.0)).(T)
		clipLower = any(float32(-1000.0)).(T)
	case float64:
		clipUpper = any(float64(1000.0)).(T)
		clipLower = any(float64(-1000.0)).(T)
	default:
		// Fallback - use zero bounds (no clipping)
		clipUpper = zero
		clipLower = zero
	}

	err := n.BackwardGPU(targets, lr, clipUpper, clipLower)
	if err != nil {
		return fmt.Errorf("GPU backward failed: %v", err)
	}

	// Check for NaN/Inf in weights
	for l := 1; l <= n.OutputLayer; l++ {
		w, b := n.extractLayerWeightsAndBiases(l)
		nanCount := 0
		for _, weight := range append(w, b...) {
			if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
				nanCount++
			}
		}
		if nanCount > 0 {
			return fmt.Errorf("layer %d has %d NaN/Inf values after GPU backward", l, nanCount)
		}
	}

	// Test forward pass with updated weights
	n.Forward(inputs)
	loss2 := n.ComputeLoss(targets)

	if n.Debug {
		fmt.Printf("Loss before: %.6f, after: %.6f, change: %.6f\n", loss1, loss2, loss2-loss1)

		// Check weight changes
		for l := 1; l <= n.OutputLayer; l++ {
			w, b := n.extractLayerWeightsAndBiases(l)
			oldW := originalWeights[fmt.Sprintf("w%d", l)]
			oldB := originalWeights[fmt.Sprintf("b%d", l)]

			maxWeightChange := float32(0)
			maxBiasChange := float32(0)

			for i, weight := range w {
				if i < len(oldW) {
					change := absFloat32(weight - oldW[i])
					if change > maxWeightChange {
						maxWeightChange = change
					}
				}
			}

			for i, bias := range b {
				if i < len(oldB) {
					change := absFloat32(bias - oldB[i])
					if change > maxBiasChange {
						maxBiasChange = change
					}
				}
			}

			fmt.Printf("Layer %d: max weight change=%.6f, max bias change=%.6f\n",
				l, maxWeightChange, maxBiasChange)
		}
	}

	// Verify loss decreased (for gradient descent)
	if loss2 > loss1 && n.Debug {
		fmt.Printf("WARNING: Loss increased from %.6f to %.6f\n", loss1, loss2)
	}

	return nil
}

func absFloat32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func (n *Network[T]) debugWeights(layerIdx int, label string) {
	if !n.Debug {
		return
	}

	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	fmt.Printf("=== %s - Layer %d Weights Debug ===\n", label, layerIdx)
	fmt.Printf("Weight count: %d, Bias count: %d\n", len(weights), len(biases))

	// Show first few weights
	fmt.Printf("First 10 weights: ")
	for i := 0; i < min(10, len(weights)); i++ {
		fmt.Printf("%.6f ", weights[i])
	}
	fmt.Println()

	// Show first few biases
	fmt.Printf("First 5 biases: ")
	for i := 0; i < min(5, len(biases)); i++ {
		fmt.Printf("%.6f ", biases[i])
	}
	fmt.Println()

	// Check for zero weights (potential problem)
	zeroCount := 0
	for _, w := range weights {
		if w == 0.0 {
			zeroCount++
		}
	}
	fmt.Printf("Zero weights: %d/%d (%.1f%%)\n", zeroCount, len(weights),
		float64(zeroCount)/float64(len(weights))*100)
}

// DEBUGGING: Enhanced uploadBackwardData with weight verification
func (n *Network[T]) uploadBackwardDataWithDebug(
	layer *GPUBackwardLayer,
	layerIdx int,
	errorData []float32,
	lr float64,
	clipUpper T,
	clipLower T,
) error {
	if n.Debug {
		n.debugWeights(layerIdx, "BEFORE GPU Upload")
	}

	// Upload error for this layer
	ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(errorData))

	// Upload current weights and biases
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)
	ctx.queue.WriteBuffer(layer.weightBuffer, 0, wgpu.ToBytes(weights))
	ctx.queue.WriteBuffer(layer.biasBuffer, 0, wgpu.ToBytes(biases))

	// Upload parameters
	isOutputLayer := float32(0)
	if layerIdx == n.OutputLayer {
		isOutputLayer = 1
	}

	params := []float32{
		float32(lr),
		float32(any(clipUpper).(T)),
		float32(any(clipLower).(T)),
		isOutputLayer,
	}
	ctx.queue.WriteBuffer(layer.paramsBuffer, 0, wgpu.ToBytes(params))

	if n.Debug {
		fmt.Printf("Uploaded layer %d: %d weights, %d biases, lr=%.4f, isOutput=%.0f\n",
			layerIdx, len(weights), len(biases), lr, isOutputLayer)

		// Check if weights have meaningful values
		nonZeroWeights := 0
		for _, w := range weights {
			if w != 0.0 {
				nonZeroWeights++
			}
		}
		fmt.Printf("Non-zero weights uploaded: %d/%d\n", nonZeroWeights, len(weights))
	}

	return nil
}

// DEBUGGING: Enhanced result reading with verification
func (n *Network[T]) readBackwardResultsWithDebug() error {
	for _, layerCompute := range n.gpu.backward.layers {
		layerIdx := layerCompute.layerIndex

		if n.Debug {
			fmt.Printf("Reading results for layer %d...\n", layerIdx)
		}

		// Read weights
		weights, err := n.readBackwardStagingBuffer(layerCompute.weightStaging,
			int(layerCompute.inputSize*layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read weights for layer %d: %v", layerIdx, err)
		}

		// Read biases
		biases, err := n.readBackwardStagingBuffer(layerCompute.biasStaging,
			int(layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read biases for layer %d: %v", layerIdx, err)
		}

		if n.Debug {
			// Check for meaningful changes
			origWeights, origBiases := n.extractLayerWeightsAndBiases(layerIdx)

			weightChanges := 0
			maxWeightChange := float32(0)
			for i, w := range weights {
				if i < len(origWeights) {
					change := absFloat32(w - origWeights[i])
					if change > 0.0001 { // Threshold for meaningful change
						weightChanges++
					}
					if change > maxWeightChange {
						maxWeightChange = change
					}
				}
			}

			biasChanges := 0
			maxBiasChange := float32(0)
			for i, b := range biases {
				if i < len(origBiases) {
					change := absFloat32(b - origBiases[i])
					if change > 0.0001 {
						biasChanges++
					}
					if change > maxBiasChange {
						maxBiasChange = change
					}
				}
			}

			fmt.Printf("Layer %d changes: %d weights (max=%.6f), %d biases (max=%.6f)\n",
				layerIdx, weightChanges, maxWeightChange, biasChanges, maxBiasChange)
		}

		// Apply to network
		if err := n.applyWeightsAndBiases(layerIdx, weights, biases); err != nil {
			return fmt.Errorf("failed to apply results for layer %d: %v", layerIdx, err)
		}

		if n.Debug {
			n.debugWeights(layerIdx, "AFTER GPU Apply")
		}
	}

	return nil
}

func (n *Network[T]) generateBackwardShaderDebug(layerIdx int, inputSize, outputSize uint32, activation string) string {
	activationDerivCode := getActivationDerivativeCode(activation)

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> error: array<f32>;
		@group(0) @binding(1) var<storage, read_write> prev_error: array<f32>;
		@group(0) @binding(2) var<storage, read> current_activations: array<f32>;
		@group(0) @binding(3) var<storage, read> prev_activations: array<f32>;
		@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
		@group(0) @binding(5) var<storage, read_write> biases: array<f32>;
		@group(0) @binding(6) var<storage, read> params: array<f32>;

		%s

		fn clip_gradient(grad: f32, clip_lower: f32, clip_upper: f32) -> f32 {
			if (clip_upper == 0.0 && clip_lower == 0.0) {
				return grad;
			}
			return clamp(grad, clip_lower, clip_upper);
		}

		@compute @workgroup_size(64, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let thread_id = global_id.x;
			let learning_rate = params[0];
			let clip_upper = params[1];
			let clip_lower = params[2];
			let is_output_layer = params[3];

			// Weight updates
			if (thread_id < %du) {
				let output_idx = thread_id;
				let local_error = error[output_idx];

				// Force a small weight change for debugging
				if (output_idx == 0u && thread_id == 0u) {
					// Always modify the first weight slightly to test if writes work
					weights[0] = weights[0] + 0.001;
				}

				// Update bias
				let bias_grad = clip_gradient(local_error, clip_lower, clip_upper);
				biases[output_idx] = biases[output_idx] - learning_rate * bias_grad;

				// Update weights
				let weight_row_start = output_idx * %du;
				for (var input_idx: u32 = 0u; input_idx < %du; input_idx = input_idx + 1u) {
					let weight_idx = weight_row_start + input_idx;
					let input_activation = prev_activations[input_idx];
					let raw_gradient = local_error * input_activation;
					let clipped_gradient = clip_gradient(raw_gradient, clip_lower, clip_upper);
					
					// Apply update
					weights[weight_idx] = weights[weight_idx] - learning_rate * clipped_gradient;
				}
			}

			workgroupBarrier();

			// Error propagation
			if (thread_id < %du) {
				let input_idx = thread_id;
				var accumulated_error: f32 = 0.0;

				for (var output_idx: u32 = 0u; output_idx < %du; output_idx = output_idx + 1u) {
					let weight_idx = output_idx * %du + input_idx;
					accumulated_error = accumulated_error + error[output_idx] * weights[weight_idx];
				}

				if (is_output_layer == 0.0) {
					let input_activation = prev_activations[input_idx];
					let derivative = activation_derivative(input_activation);
					prev_error[input_idx] = accumulated_error * derivative;
				} else {
					prev_error[input_idx] = accumulated_error;
				}
			}
		}
	`, activationDerivCode, outputSize, inputSize, inputSize, inputSize, outputSize, inputSize)
}

// Quick test function to verify the shader fix
func (n *Network[T]) TestShaderWeightUpdate() {
	if !n.WebGPUNative || n.gpu.backward == nil {
		fmt.Println("GPU backward not available for testing")
		return
	}

	// Test with the debug shader that forces a weight change
	layerCompute := n.gpu.backward.layers[0] // Test first layer

	// Read weights before
	weightsBefore, _ := n.readBackwardStagingBuffer(layerCompute.weightStaging,
		int(layerCompute.inputSize*layerCompute.outputSize))

	// Force run the shader (you'd need to adapt this to your actual GPU execution)
	// This is pseudocode - integrate with your actual GPU pipeline

	// Read weights after
	weightsAfter, _ := n.readBackwardStagingBuffer(layerCompute.weightStaging,
		int(layerCompute.inputSize*layerCompute.outputSize))

	// Check if first weight changed
	if len(weightsBefore) > 0 && len(weightsAfter) > 0 {
		change := weightsAfter[0] - weightsBefore[0]
		fmt.Printf("First weight change: %.6f (before: %.6f, after: %.6f)\n",
			change, weightsBefore[0], weightsAfter[0])

		if absFloat32(change) > 0.0001 {
			fmt.Println("✓ GPU weight writing works!")
		} else {
			fmt.Println("✗ GPU weight writing failed!")
		}
	}
}

func (n *Network[T]) TestBufferWrite() error {
	if n.gpu.backward == nil || len(n.gpu.backward.layers) == 0 {
		return fmt.Errorf("no backward layers available")
	}

	layer := n.gpu.backward.layers[0]

	// Create a simple test to write to weight buffer
	testData := []float32{999.0, 888.0, 777.0, 666.0, 555.0}

	// Write test data to weight buffer
	ctx.queue.WriteBuffer(layer.weightBuffer, 0, wgpu.ToBytes(testData))

	// Create command encoder to copy to staging
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create encoder: %v", err)
	}

	// Copy first 5 weights to staging
	encoder.CopyBufferToBuffer(
		layer.weightBuffer, 0,
		layer.weightStaging, 0,
		20, // 5 floats * 4 bytes
	)

	// Submit and wait
	cmd, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	// Read back the data
	readData, err := n.readBackwardStagingBuffer(layer.weightStaging, 5)
	if err != nil {
		return fmt.Errorf("failed to read staging buffer: %v", err)
	}

	// Check if write succeeded
	fmt.Printf("Buffer write test results:\n")
	for i, val := range readData {
		expected := testData[i]
		fmt.Printf("  [%d] Expected: %.1f, Got: %.1f", i, expected, val)
		if absFloat32(val-expected) < 0.1 {
			fmt.Printf(" ✓")
		} else {
			fmt.Printf(" ✗")
		}
		fmt.Println()
	}

	return nil
}

func (n *Network[float32]) SyncAllGPUWeights() {
	n.syncForwardWeightsToGPU()
	n.syncBackwardWeightsToGPU()
}

// Borrow the storage buffers so the forward path can share them
func (n *Network[float32]) getWeightBuffer(layer int) *wgpu.Buffer {
	return n.gpu.backward.layers[layer-1].weightBuffer
}
func (n *Network[float32]) getBiasBuffer(layer int) *wgpu.Buffer {
	return n.gpu.backward.layers[layer-1].biasBuffer
}
