// webgpu_optimized.go - Fixed version
package paragon

import (
	"fmt"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

// GPULayerCompute represents a single layer's GPU computation
type GPULayerCompute struct {
	pipeline        *wgpu.ComputePipeline
	bindGroup       *wgpu.BindGroup
	bindGroupLayout *wgpu.BindGroupLayout
	inputBuffer     *wgpu.Buffer
	outputBuffer    *wgpu.Buffer
	weightBuffer    *wgpu.Buffer
	biasBuffer      *wgpu.Buffer
	stagingBuffer   *wgpu.Buffer
	workgroupsX     uint32
	workgroupsY     uint32
	inputSize       uint32
	outputSize      uint32
	layerIndex      int

	// Training additions
	backwardPipeline   *wgpu.ComputePipeline
	backwardBindGroup  *wgpu.BindGroup
	backwardLayout     *wgpu.BindGroupLayout
	errorInputBuffer   *wgpu.Buffer
	errorOutputBuffer  *wgpu.Buffer
	weightGradBuffer   *wgpu.Buffer
	biasGradBuffer     *wgpu.Buffer
	paramsBuffer       *wgpu.Buffer
	errorStagingBuffer *wgpu.Buffer
	gradStagingBuffer  *wgpu.Buffer
}

// GPUTrainingCompute extends GPULayerCompute for backward pass
type GPUTrainingCompute struct {
	*GPULayerCompute
}

// GPUCompute manages optimized GPU neural network computation
type GPUCompute struct {
	device      *wgpu.Device
	queue       *wgpu.Queue
	layers      []*GPULayerCompute
	initialized bool
	debug       bool

	// Training extensions
	trainingLayers  []*GPUTrainingCompute
	trainingEnabled bool
}

// GPUComputeEnhanced is an alias for GPUCompute for compatibility
type GPUComputeEnhanced = GPUCompute

// TrainingParameters for GPU training
type TrainingParams struct {
	LearningRate float32
	ClipUpper    float32
	ClipLower    float32
	LayerIndex   uint32
}

// Training state management
type GPUTrainingState struct {
	epoch           int
	batch           int
	currentLoss     float32
	learningRate    float32
	clippingEnabled bool
	clipUpper       float32
	clipLower       float32
}

// Initialize GPU compute for the network
func (n *Network[T]) InitializeOptimizedGPU() error {
	// Only support float32 for now
	if any(*new(T)).(T) != T(float32(0)) {
		return fmt.Errorf("WebGPU only supports float32 networks currently")
	}

	// Initialize WebGPU context
	if err := ensureGPU(); err != nil {
		return fmt.Errorf("WebGPU initialization failed: %v", err)
	}

	// Clear any existing GPU state
	n.CleanupOptimizedGPU()

	// Initialize optimized GPU compute
	n.gpu.optimized = &GPUComputeEnhanced{
		device: ctx.device,
		queue:  ctx.queue,
		debug:  n.Debug,
	}

	// Create layer-specific compute pipelines
	for l := 1; l <= n.OutputLayer; l++ {
		layerCompute, err := n.createLayerCompute(l)
		if err != nil {
			n.CleanupOptimizedGPU()
			return fmt.Errorf("failed to create compute for layer %d: %v", l, err)
		}
		n.gpu.optimized.layers = append(n.gpu.optimized.layers, layerCompute)
	}

	n.gpu.optimized.initialized = true
	n.WebGPUNative = true

	if n.Debug {
		fmt.Printf("✅ Optimized WebGPU initialized with %d layers\n", len(n.gpu.optimized.layers))
	}

	return nil
}

// Initialize GPU training for the network
func (n *Network[T]) InitializeGPUTraining() error {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return fmt.Errorf("base GPU compute not initialized")
	}

	// Use hybrid mode for now (GPU forward, CPU backward)
	// This avoids atomic operations issues while providing GPU acceleration
	n.gpu.optimized.trainingEnabled = true

	if n.Debug {
		fmt.Printf("✅ GPU training initialized in hybrid mode (GPU forward, CPU backward)\n")
	}

	return nil
}

// Create GPU compute resources for a single layer
func (n *Network[T]) createLayerCompute(layerIdx int) (*GPULayerCompute, error) {
	if layerIdx <= 0 || layerIdx >= len(n.Layers) {
		return nil, fmt.Errorf("invalid layer index: %d", layerIdx)
	}

	prevLayer := n.Layers[layerIdx-1]
	currentLayer := n.Layers[layerIdx]

	inputSize := uint32(prevLayer.Width * prevLayer.Height)
	outputSize := uint32(currentLayer.Width * currentLayer.Height)

	if inputSize == 0 || outputSize == 0 {
		return nil, fmt.Errorf("invalid layer dimensions: input=%d, output=%d", inputSize, outputSize)
	}

	// Create shader for this specific layer
	shaderCode := n.generateLayerShader(layerIdx, inputSize, outputSize)

	// Create shader module
	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create shader module: %v", err)
	}

	// Create bind group layout
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
		},
	})
	if err != nil {
		module.Release()
		return nil, fmt.Errorf("failed to create bind group layout: %v", err)
	}

	// Create pipeline layout
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		module.Release()
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create pipeline layout: %v", err)
	}

	// Create compute pipeline
	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		module.Release()
		bindGroupLayout.Release()
		pipelineLayout.Release()
		return nil, fmt.Errorf("failed to create compute pipeline: %v", err)
	}

	// Clean up intermediate resources
	module.Release()
	pipelineLayout.Release()

	// Initialize layer compute structure
	layerCompute := &GPULayerCompute{
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
		inputSize:       inputSize,
		outputSize:      outputSize,
		layerIndex:      layerIdx,
	}

	// Create buffers with proper error handling
	if err := n.createLayerBuffers(layerCompute); err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create buffers: %v", err)
	}

	// Create bind group
	layerCompute.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_BindGroup", layerIdx),
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: layerCompute.inputBuffer, Size: layerCompute.inputBuffer.GetSize()},
			{Binding: 1, Buffer: layerCompute.outputBuffer, Size: layerCompute.outputBuffer.GetSize()},
			{Binding: 2, Buffer: layerCompute.weightBuffer, Size: layerCompute.weightBuffer.GetSize()},
			{Binding: 3, Buffer: layerCompute.biasBuffer, Size: layerCompute.biasBuffer.GetSize()},
		},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bind group: %v", err)
	}

	// Calculate optimal workgroup dimensions
	layerCompute.workgroupsX = (outputSize + 255) / 256
	layerCompute.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created layer %d compute: %dx%d -> %dx%d, workgroups: %d\n",
			layerIdx, prevLayer.Width, prevLayer.Height,
			currentLayer.Width, currentLayer.Height, layerCompute.workgroupsX)
	}

	return layerCompute, nil
}

func (n *Network[T]) createLayerBuffers(lc *GPULayerCompute) error {
	var err error

	// Input buffer
	lc.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Input", lc.layerIndex),
		Size:  uint64(lc.inputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create input buffer: %v", err)
	}

	// Output buffer
	lc.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Output", lc.layerIndex),
		Size:  uint64(lc.outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Staging buffer for CPU readback
	lc.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Staging", lc.layerIndex),
		Size:  uint64(lc.outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create staging buffer: %v", err)
	}

	// Prepare weight and bias data
	weights, biases := n.extractLayerWeightsAndBiases(lc.layerIndex)

	// Weight buffer
	lc.weightBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Weights", lc.layerIndex),
		Contents: wgpu.ToBytes(weights),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create weight buffer: %v", err)
	}

	// Bias buffer
	lc.biasBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Biases", lc.layerIndex),
		Contents: wgpu.ToBytes(biases),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create bias buffer: %v", err)
	}

	return nil
}

// Helper method to clean up a layer's resources
func (lc *GPULayerCompute) cleanup() {
	// Forward pass cleanup
	if lc.inputBuffer != nil {
		lc.inputBuffer.Destroy()
	}
	if lc.outputBuffer != nil {
		lc.outputBuffer.Destroy()
	}
	if lc.weightBuffer != nil {
		lc.weightBuffer.Destroy()
	}
	if lc.biasBuffer != nil {
		lc.biasBuffer.Destroy()
	}
	if lc.stagingBuffer != nil {
		lc.stagingBuffer.Destroy()
	}

	// Training cleanup
	if lc.errorInputBuffer != nil {
		lc.errorInputBuffer.Destroy()
	}
	if lc.errorOutputBuffer != nil {
		lc.errorOutputBuffer.Destroy()
	}
	if lc.weightGradBuffer != nil {
		lc.weightGradBuffer.Destroy()
	}
	if lc.biasGradBuffer != nil {
		lc.biasGradBuffer.Destroy()
	}
	if lc.paramsBuffer != nil {
		lc.paramsBuffer.Destroy()
	}
	if lc.errorStagingBuffer != nil {
		lc.errorStagingBuffer.Destroy()
	}
	if lc.gradStagingBuffer != nil {
		lc.gradStagingBuffer.Destroy()
	}

	// Release bind groups and pipelines
	if lc.bindGroup != nil {
		lc.bindGroup.Release()
	}
	if lc.backwardBindGroup != nil {
		lc.backwardBindGroup.Release()
	}
	if lc.bindGroupLayout != nil {
		lc.bindGroupLayout.Release()
	}
	if lc.backwardLayout != nil {
		lc.backwardLayout.Release()
	}
	if lc.pipeline != nil {
		lc.pipeline.Release()
	}
	if lc.backwardPipeline != nil {
		lc.backwardPipeline.Release()
	}
}

// Generate optimized shader for a specific layer
func (n *Network[T]) generateLayerShader(layerIdx int, inputSize, outputSize uint32) string {
	currentLayer := n.Layers[layerIdx]
	activation := currentLayer.Neurons[0][0].Activation
	typ := n.gpu.wgslType
	activationCode := getActivationCode(activation, typ)

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input: array<%s>;
		@group(0) @binding(1) var<storage, read_write> output: array<%s>;
		@group(0) @binding(2) var<storage, read> weights: array<%s>;
		@group(0) @binding(3) var<storage, read> biases: array<%s>;

		%s

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let output_idx = global_id.x;

			if (output_idx >= %du) {
				return;
			}

			var sum: %s = biases[output_idx];
			let weight_offset = output_idx * %du;
			for (var i: u32 = 0u; i < %du; i++) {
				sum += weights[weight_offset + i] * input[i];
			}

			output[output_idx] = activate(sum);
		}
	`, typ, typ, typ, typ, activationCode, outputSize, typ, inputSize, inputSize)
}

// Get activation function WGSL code
func getActivationCode(activation, typ string) string {
	switch activation {
	case "relu":
		return fmt.Sprintf(`fn activate(x: %s) -> %s { return max(%s(0), x); }`, typ, typ, typ)

	case "leaky_relu":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 { return select(0.01 * x, x, x > 0.0); }`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 { return x; }`
		}
		// Pure integer leaky ReLU matching CPU
		return `fn activate(x: i32) -> i32 {
			if (x >= i32(0)) { 
				return x; 
			}
			// Pure integer 1% leak: x / 100
			var leak = x / i32(100);
			if (leak == i32(0) && x < i32(0)) {
				leak = i32(-1); // Minimum leak for negative values
			}
			return leak;
		}`

	case "elu":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 {
				if (x >= 0.0) { return x; }
				return exp(max(x, -10.0)) - 1.0;
			}`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 { return x; }`
		}
		// Pure integer ELU approximation matching CPU
		return `fn activate(x: i32) -> i32 {
			if (x >= i32(0)) { 
				return x; 
			}
			let scale = i32(2147483647);
			if (x <= -scale) {
				return -scale; // Cap at -1.0 in fixed point
			}
			// Simple approximation: ELU(x) ≈ x/2 for negative x
			return x / i32(2);
		}`

	case "tanh":
		const tanhApprox = `
fn tanh_approx(input: f32) -> f32 {
    if (input >  1.0) { return  1.0; }
    if (input < -1.0) { return -1.0; }
    if (input >= 0.0) {
        if (input < 0.25) { return input; }
        let denom = 1.0 + input * 2.0;
        return 1.0 - 2.0 / denom;
    } else {
        if (input > -0.25) { return input; }
        let abs_input = -input;
        let denom = 1.0 + abs_input * 2.0;
        return -1.0 + 2.0 / denom;
    }
}
`

		if typ == "f32" {
			return tanhApprox + `
fn activate(x: f32) -> f32 {
    return tanh_approx(x);
}`
		}

		if typ == "i32" || typ == "int32" {
			return tanhApprox + `
fn activate(x: i32) -> i32 {
    return i32(tanh_approx(f32(x)));
}`
		}

		if typ == "u32" || typ == "uint32" {
			return tanhApprox + `
fn activate(x: u32) -> u32 {
    let t = tanh_approx(f32(x));
    return u32(max(t, 0.0));
}`
		}

		return fmt.Sprintf(`fn activate(x: %s) -> %s { return x; }`, typ, typ)

	case "sigmoid":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 {
				let scaled = f32(x) / f32(2147483647);
				let s = 1.0 / (1.0 + exp(-scaled * 0.5));
				return u32(round(s * f32(2147483647) * 2.0));
			}`
		}
		return `fn activate(x: i32) -> i32 {
			let scaled = f32(x) / f32(2147483647);
			let s = 1.0 / (1.0 + exp(-scaled));
			return i32(round(s * f32(2147483647)));
		}`

	default:
		return fmt.Sprintf(`fn activate(x: %s) -> %s { return x; }`, typ, typ)
	}
}

// Extract weights and biases for a specific layer
func (n *Network[T]) extractLayerWeightsAndBiases(layerIdx int) ([]T, []T) {
	currentLayer := n.Layers[layerIdx]
	prevLayer := n.Layers[layerIdx-1]

	inputSize := prevLayer.Width * prevLayer.Height
	outputSize := currentLayer.Width * currentLayer.Height

	weights := make([]T, outputSize*inputSize)
	biases := make([]T, outputSize)

	// Debug: Print layer info
	if n.Debug {
		fmt.Printf("Extracting weights for layer %d: input=%d, output=%d\n",
			layerIdx, inputSize, outputSize)
	}

	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			// Extract bias
			biases[neuronIdx] = neuron.Bias

			// CRITICAL FIX: Extract weights in the same order as CPU connections
			weightOffset := neuronIdx * inputSize

			// Create a map to ensure correct weight ordering
			weightMap := make(map[int]T)

			for _, conn := range neuron.Inputs {
				// Calculate the linear index in the same way as CPU
				srcIdx := conn.SourceY*prevLayer.Width + conn.SourceX
				if srcIdx < inputSize {
					weightMap[srcIdx] = conn.Weight
				}
			}

			// Fill weights array in proper order
			for i := 0; i < inputSize; i++ {
				if weight, exists := weightMap[i]; exists {
					weights[weightOffset+i] = weight
				} else {
					// This shouldn't happen in a fully connected layer
					weights[weightOffset+i] = T(0)
					if n.Debug {
						fmt.Printf("Warning: missing weight for neuron %d, input %d\n", neuronIdx, i)
					}
				}
			}
		}
	}

	if n.Debug {
		fmt.Printf("Extracted %d weights and %d biases\n", len(weights), len(biases))
	}

	return weights, biases
}

// Optimized forward pass using proper GPU parallelization
func (n *Network[T]) ForwardGPUOptimized(inputs [][]float64) error {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return fmt.Errorf("optimized GPU not initialized")
	}

	if len(n.gpu.optimized.layers) == 0 {
		return fmt.Errorf("no GPU layers available")
	}

	// Convert input data to float32
	inputData := make([]float32, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, float32(val))
		}
	}

	// Set input layer values (matching CPU behavior)
	inputLayer := &n.Layers[n.InputLayer]
	idx := 0
	for y := 0; y < inputLayer.Height; y++ {
		for x := 0; x < inputLayer.Width; x++ {
			if idx < len(inputData) {
				inputLayer.Neurons[y][x].Value = T(inputData[idx])
				idx++
			}
		}
	}

	// Process layers sequentially with proper synchronization
	for i, layerCompute := range n.gpu.optimized.layers {
		if err := n.processGPULayer(i, layerCompute, inputData); err != nil {
			return fmt.Errorf("failed to process layer %d: %v", i+1, err)
		}

		// Update inputData for next layer
		if i < len(n.gpu.optimized.layers)-1 {
			layerOutput, err := n.readLayerOutput(layerCompute)
			if err != nil {
				return fmt.Errorf("failed to read layer %d output: %v", i+1, err)
			}
			inputData = layerOutput
		}
	}

	// Read final output and apply to network
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readLayerOutput(finalLayer)
	if err != nil {
		return fmt.Errorf("failed to read final output: %v", err)
	}

	// Apply final output to network neurons
	n.applyFinalOutputFixed(finalOutput)

	return nil
}

// Read layer output with improved synchronization
func (n *Network[T]) readLayerOutput(lc *GPULayerCompute) ([]float32, error) {
	done := make(chan wgpu.BufferMapAsyncStatus, 1)

	err := lc.stagingBuffer.MapAsync(wgpu.MapModeRead, 0, lc.stagingBuffer.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Wait for mapping with timeout
	timeout := time.After(5 * time.Second)
	for {
		ctx.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("buffer mapping failed: %v", status)
			}
			goto readData
		case <-timeout:
			return nil, fmt.Errorf("buffer mapping timeout")
		default:
			time.Sleep(time.Microsecond * 100)
		}
	}

readData:
	data := lc.stagingBuffer.GetMappedRange(0, uint(lc.outputSize*4))
	if data == nil {
		lc.stagingBuffer.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	result := make([]float32, lc.outputSize)
	copy(result, wgpu.FromBytes[float32](data))
	lc.stagingBuffer.Unmap()

	return result, nil
}

// Fixed output application
func (n *Network[T]) applyFinalOutputFixed(output []float32) {
	outputLayer := &n.Layers[n.OutputLayer]
	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			if idx < len(output) {
				outputLayer.Neurons[y][x].Value = T(output[idx])
				idx++
			}
		}
	}

	// Apply softmax if needed
	if len(outputLayer.Neurons) > 0 && len(outputLayer.Neurons[0]) > 0 &&
		outputLayer.Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}

	if n.Debug {
		fmt.Printf("Applied %d values to output layer\n", len(output))
	}
}

func (n *Network[T]) applyLayerOutput(layerIdx int, output []T) {
	if layerIdx < 0 || layerIdx >= len(n.Layers) {
		return
	}

	layer := &n.Layers[layerIdx]
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			if idx < len(output) {
				layer.Neurons[y][x].Value = output[idx]
				idx++
			}
		}
	}

	if n.Debug {
		fmt.Printf("Applied %d values to layer %d\n", len(output), layerIdx)
	}
}

func (n *Network[T]) processGPULayer(layerIndex int, lc *GPULayerCompute, inputData []float32) error {
	// Create command encoder
	encoder, err := ctx.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: fmt.Sprintf("Layer_%d_Commands", layerIndex+1),
	})
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Upload input data
	ctx.queue.WriteBuffer(lc.inputBuffer, 0, wgpu.ToBytes(inputData))

	// Create compute pass
	computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: fmt.Sprintf("Layer_%d_Compute", layerIndex+1),
	})

	computePass.SetPipeline(lc.pipeline)
	computePass.SetBindGroup(0, lc.bindGroup, nil)
	computePass.DispatchWorkgroups(lc.workgroupsX, lc.workgroupsY, 1)
	computePass.End()

	// Copy output to staging buffer
	encoder.CopyBufferToBuffer(
		lc.outputBuffer, 0,
		lc.stagingBuffer, 0,
		uint64(lc.outputSize)*4,
	)

	// Submit commands
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)

	// Wait for completion with timeout
	start := time.Now()
	for time.Since(start) < time.Second*5 {
		ctx.device.Poll(true, nil)
		time.Sleep(time.Millisecond)
	}

	return nil
}

// Read data from staging buffer with proper synchronization
func (n *Network[T]) readStagingBuffer(buffer *wgpu.Buffer, size int) ([]T, error) {
	done := make(chan wgpu.BufferMapAsyncStatus, 1)

	err := buffer.MapAsync(wgpu.MapModeRead, 0, buffer.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Poll until mapped with timeout
	timeout := 0
	for {
		ctx.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("buffer mapping failed: %v", status)
			}
			goto readData
		default:
			timeout++
			if timeout > 10000 { // Prevent infinite loop
				return nil, fmt.Errorf("buffer mapping timeout")
			}
			time.Sleep(time.Microsecond * 10)
		}
	}

readData:
	data := buffer.GetMappedRange(0, uint(size*4))
	if data == nil {
		buffer.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	// Convert from float32 to T
	floatData := wgpu.FromBytes[float32](data)
	result := make([]T, size)
	for i, v := range floatData {
		if i < len(result) {
			result[i] = T(v)
		}
	}
	buffer.Unmap()

	return result, nil
}

// Apply final GPU output to network neurons
func (n *Network[T]) applyFinalOutput(output []T) {
	outputLayer := &n.Layers[n.OutputLayer]
	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			if idx < len(output) {
				outputLayer.Neurons[y][x].Value = output[idx]
				idx++
			}
		}
	}

	// Apply softmax if needed
	if len(outputLayer.Neurons) > 0 && len(outputLayer.Neurons[0]) > 0 &&
		outputLayer.Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// Update the main Forward method to use optimized GPU
func (n *Network[T]) Forward(inputs [][]float64) {
	// Use optimized GPU if available and enabled
	if n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized {
		err := n.ForwardGPUOptimized(inputs)
		if err != nil {
			if n.Debug {
				fmt.Printf("⚠️ Optimized GPU forward failed, falling back to CPU: %v\n", err)
			}
			// Fall back to CPU
			n.forwardCPU(inputs)
		}
		return
	}

	// Fallback to existing implementation
	n.forwardCPU(inputs)
}

// Train a single sample using hybrid GPU/CPU approach
func (n *Network[T]) TrainSampleGPU(
	input [][]float64,
	target [][]float64,
	learningRate float64,
	clipUpper T,
	clipLower T,
) error {
	// Use hybrid approach: GPU forward pass + CPU backward pass
	// This avoids the atomic operations issues while still getting GPU acceleration

	// GPU forward pass
	if n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized {
		err := n.ForwardGPUOptimized(input)
		if err != nil {
			if n.Debug {
				fmt.Printf("⚠️ GPU forward failed, using CPU: %v\n", err)
			}
			// Fallback to CPU forward if GPU fails
			n.Forward(input)
		}
	} else {
		// CPU forward pass
		n.Forward(input)
	}

	// CPU backward pass (more reliable than GPU for now)
	n.Backward(target, learningRate, clipUpper, clipLower)

	return nil
}

// Monitoring and debugging functions
func (n *Network[T]) GetGPUTrainingStats() map[string]interface{} {
	if n.gpu.optimized == nil {
		return map[string]interface{}{
			"gpu_enabled": false,
		}
	}

	stats := map[string]interface{}{
		"gpu_enabled":      n.gpu.optimized.initialized,
		"training_mode":    "hybrid", // GPU forward, CPU backward
		"num_layers":       len(n.gpu.optimized.layers),
		"training_enabled": n.gpu.optimized.trainingEnabled,
	}

	return stats
}

// Clean up GPU resources
func (n *Network[T]) CleanupOptimizedGPU() {
	if n.gpu.optimized == nil {
		return
	}

	for i, layer := range n.gpu.optimized.layers {
		layer.cleanup()
		if n.Debug {
			fmt.Printf("Cleaned up GPU resources for layer %d\n", i)
		}
	}

	n.gpu.optimized.layers = nil
	n.gpu.optimized.trainingLayers = nil
	n.gpu.optimized.initialized = false
	n.gpu.optimized.trainingEnabled = false
	n.gpu.optimized = nil
}

func (n *Network[T]) IsGPUAvailable() bool {
	return n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized
}

func (n *Network[T]) GetGPUStatus() string {
	if !n.WebGPUNative {
		return "disabled"
	}

	if n.gpu.optimized == nil {
		return "not_initialized"
	}

	if !n.gpu.optimized.initialized {
		return "initialization_failed"
	}

	return "active"
}
