// webgpu_optimized.go
package paragon

import (
	"fmt"

	"github.com/rajveermalviya/go-webgpu/wgpu"
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
}

// GPUCompute manages optimized GPU neural network computation
type GPUCompute struct {
	device      *wgpu.Device
	queue       *wgpu.Queue
	layers      []*GPULayerCompute
	initialized bool
	debug       bool
}

// Initialize GPU compute for the network
func (n *Network[T]) InitializeOptimizedGPU() error {
	if any(*new(T)).(T) != T(float32(0)) {
		return fmt.Errorf("GPU acceleration only supported for float32 networks")
	}

	ensureGPU()

	n.gpu.optimized = &GPUCompute{
		device: ctx.device,
		queue:  ctx.queue,
		debug:  n.Debug,
	}

	// Create layer-specific compute pipelines
	for l := 1; l <= n.OutputLayer; l++ {
		layerCompute, err := n.createLayerCompute(l)
		if err != nil {
			return fmt.Errorf("failed to create compute for layer %d: %v", l, err)
		}
		n.gpu.optimized.layers = append(n.gpu.optimized.layers, layerCompute)
	}

	n.gpu.optimized.initialized = true
	return nil
}

// Create GPU compute resources for a single layer
func (n *Network[T]) createLayerCompute(layerIdx int) (*GPULayerCompute, error) {
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
	defer module.Release() // Clean up after pipeline creation

	// Create bind group layout first
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_Storage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingType_ReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bind group layout: %v", err)
	}

	// Create pipeline layout
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create pipeline layout: %v", err)
	}
	defer pipelineLayout.Release() // Clean up after pipeline creation

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
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create compute pipeline: %v", err)
	}

	// Create buffers
	layerCompute := &GPULayerCompute{
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
		inputSize:       inputSize,
		outputSize:      outputSize,
		layerIndex:      layerIdx,
	}

	// Input buffer
	layerCompute.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Input", layerIdx),
		Size:  uint64(inputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	// Output buffer
	layerCompute.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Output", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Staging buffer for CPU readback
	layerCompute.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Staging", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create staging buffer: %v", err)
	}

	// Prepare weight and bias data
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	// Weight buffer
	layerCompute.weightBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Weights", layerIdx),
		Contents: wgpu.ToBytes(weights),
		Usage:    wgpu.BufferUsage_Storage,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create weight buffer: %v", err)
	}

	// Bias buffer
	layerCompute.biasBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Biases", layerIdx),
		Contents: wgpu.ToBytes(biases),
		Usage:    wgpu.BufferUsage_Storage,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bias buffer: %v", err)
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
	layerCompute.workgroupsX = (outputSize + 255) / 256 // 256 threads per workgroup
	layerCompute.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created layer %d compute: %dx%d -> %dx%d, workgroups: %d\n",
			layerIdx, prevLayer.Width, prevLayer.Height,
			currentLayer.Width, currentLayer.Height, layerCompute.workgroupsX)
	}

	return layerCompute, nil
}

// Helper method to clean up a layer's resources
func (lc *GPULayerCompute) cleanup() {
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
	if lc.bindGroup != nil {
		lc.bindGroup.Release()
	}
	if lc.bindGroupLayout != nil {
		lc.bindGroupLayout.Release()
	}
	if lc.pipeline != nil {
		lc.pipeline.Release()
	}
}

// Generate optimized shader for a specific layer
func (n *Network[T]) generateLayerShader(layerIdx int, inputSize, outputSize uint32) string {
	currentLayer := n.Layers[layerIdx]
	activation := currentLayer.Neurons[0][0].Activation
	activationCode := getActivationCode(activation)

	// Use specialized matrix multiplication shader
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input: array<f32>;
		@group(0) @binding(1) var<storage, read_write> output: array<f32>;
		@group(0) @binding(2) var<storage, read> weights: array<f32>;
		@group(0) @binding(3) var<storage, read> biases: array<f32>;

		%s

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let output_idx = global_id.x;
			
			if (output_idx >= %du) {
				return;
			}

			var sum: f32 = biases[output_idx];
			
			// Optimized matrix-vector multiplication
			let weight_offset = output_idx * %du;
			for (var i: u32 = 0u; i < %du; i++) {
				sum += weights[weight_offset + i] * input[i];
			}
			
			// Apply activation function
			output[output_idx] = activate(sum);
		}
	`, activationCode, outputSize, inputSize, inputSize)
}

// Get activation function WGSL code
func getActivationCode(activation string) string {
	switch activation {
	case "relu":
		return `fn activate(x: f32) -> f32 { return max(0.0, x); }`
	case "leaky_relu":
		return `fn activate(x: f32) -> f32 { return select(0.01 * x, x, x > 0.0); }`
	case "sigmoid":
		return `fn activate(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`
	case "tanh":
		return `fn activate(x: f32) -> f32 { return tanh(x); }`
	default:
		return `fn activate(x: f32) -> f32 { return x; }`
	}
}

// Extract weights and biases for a specific layer
func (n *Network[T]) extractLayerWeightsAndBiases(layerIdx int) ([]float32, []float32) {
	currentLayer := n.Layers[layerIdx]
	prevLayer := n.Layers[layerIdx-1]

	inputSize := prevLayer.Width * prevLayer.Height
	outputSize := currentLayer.Width * currentLayer.Height

	weights := make([]float32, outputSize*inputSize)
	biases := make([]float32, outputSize)

	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			// Extract bias
			biases[neuronIdx] = float32(any(neuron.Bias).(T))

			// Extract weights in proper order for matrix multiplication
			weightOffset := neuronIdx * inputSize
			for i, conn := range neuron.Inputs {
				if i < inputSize {
					weights[weightOffset+i] = float32(any(conn.Weight).(T))
				}
			}
		}
	}

	return weights, biases
}

// Optimized forward pass using proper GPU parallelization
func (n *Network[T]) ForwardGPUOptimized(inputs [][]float64) error {
	if !n.gpu.optimized.initialized {
		return fmt.Errorf("optimized GPU not initialized")
	}

	// Convert input to float32
	inputData := make([]float32, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, float32(val))
		}
	}

	// Create command encoder
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Process each layer sequentially but with full GPU parallelization within each layer
	var currentInput []float32 = inputData

	for i, layerCompute := range n.gpu.optimized.layers {
		// Upload input data to GPU
		ctx.queue.WriteBuffer(layerCompute.inputBuffer, 0, wgpu.ToBytes(currentInput))

		// Create compute pass for this layer
		computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Compute", i+1),
		})

		computePass.SetPipeline(layerCompute.pipeline)
		computePass.SetBindGroup(0, layerCompute.bindGroup, nil)

		// Dispatch with optimal workgroup size
		computePass.DispatchWorkgroups(layerCompute.workgroupsX, layerCompute.workgroupsY, 1)
		computePass.End()

		// Copy output to staging buffer for readback (only for final layer or if debugging)
		if i == len(n.gpu.optimized.layers)-1 || n.Debug {
			encoder.CopyBufferToBuffer(
				layerCompute.outputBuffer, 0,
				layerCompute.stagingBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}

		// For next iteration, we need to read the output as input
		// This is the bottleneck we need to minimize
		if i < len(n.gpu.optimized.layers)-1 {
			nextLayerCompute := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				layerCompute.outputBuffer, 0,
				nextLayerCompute.inputBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}
	}

	// Submit all commands at once
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)

	// Wait for completion and read final results
	ctx.device.Poll(true, nil)

	// Read final layer output
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readStagingBuffer(finalLayer.stagingBuffer, int(finalLayer.outputSize))
	if err != nil {
		return fmt.Errorf("failed to read final output: %v", err)
	}

	// Apply results to network
	n.applyFinalOutput(finalOutput)

	return nil
}

// Read data from staging buffer with proper synchronization
func (n *Network[T]) readStagingBuffer(buffer *wgpu.Buffer, size int) ([]float32, error) {
	done := make(chan wgpu.BufferMapAsyncStatus, 1)

	err := buffer.MapAsync(wgpu.MapMode_Read, 0, buffer.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Poll until mapped
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

// Apply final GPU output to network neurons
func (n *Network[T]) applyFinalOutput(output []float32) {
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
	if outputLayer.Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// Pipeline GPU computation using double buffering to reduce CPU-GPU sync
func (n *Network[T]) ForwardGPUPipelined(inputs [][]float64) error {
	if len(n.gpu.optimized.layers) < 2 {
		return n.ForwardGPUOptimized(inputs) // Fall back to sequential
	}

	// Convert input
	inputData := make([]float32, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, float32(val))
		}
	}

	// Create multiple command encoders for pipelining
	encoders := make([]*wgpu.CommandEncoder, 0)

	// First layer
	encoder1, _ := ctx.device.CreateCommandEncoder(nil)
	encoders = append(encoders, encoder1)

	layer0 := n.gpu.optimized.layers[0]
	ctx.queue.WriteBuffer(layer0.inputBuffer, 0, wgpu.ToBytes(inputData))

	pass1 := encoder1.BeginComputePass(nil)
	pass1.SetPipeline(layer0.pipeline)
	pass1.SetBindGroup(0, layer0.bindGroup, nil)
	pass1.DispatchWorkgroups(layer0.workgroupsX, layer0.workgroupsY, 1)
	pass1.End()

	// Copy to next layer's input
	if len(n.gpu.optimized.layers) > 1 {
		nextLayer := n.gpu.optimized.layers[1]
		encoder1.CopyBufferToBuffer(
			layer0.outputBuffer, 0,
			nextLayer.inputBuffer, 0,
			uint64(layer0.outputSize)*4,
		)
	}

	// Submit first layer
	cmd1, _ := encoder1.Finish(nil)
	ctx.queue.Submit(cmd1)

	// Process remaining layers with overlapping
	for i := 1; i < len(n.gpu.optimized.layers); i++ {
		encoder, _ := ctx.device.CreateCommandEncoder(nil)
		layer := n.gpu.optimized.layers[i]

		pass := encoder.BeginComputePass(nil)
		pass.SetPipeline(layer.pipeline)
		pass.SetBindGroup(0, layer.bindGroup, nil)
		pass.DispatchWorkgroups(layer.workgroupsX, layer.workgroupsY, 1)
		pass.End()

		// Copy to staging for final layer
		if i == len(n.gpu.optimized.layers)-1 {
			encoder.CopyBufferToBuffer(
				layer.outputBuffer, 0,
				layer.stagingBuffer, 0,
				uint64(layer.outputSize)*4,
			)
		} else {
			// Copy to next layer
			nextLayer := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				layer.outputBuffer, 0,
				nextLayer.inputBuffer, 0,
				uint64(layer.outputSize)*4,
			)
		}

		cmd, _ := encoder.Finish(nil)
		ctx.queue.Submit(cmd)
	}

	// Wait and read final results
	ctx.device.Poll(true, nil)
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readStagingBuffer(finalLayer.stagingBuffer, int(finalLayer.outputSize))
	if err != nil {
		return err
	}

	n.applyFinalOutput(finalOutput)
	return nil
}

// Add this to your Network struct's gpu field:
/*
type gpuResources struct {
	// ... existing fields ...
	optimized *GPUCompute  // Add this field
}
*/

// Update the main Forward method to use optimized GPU
func (n *Network[T]) Forward(inputs [][]float64) {
	// Use optimized GPU if available and enabled
	if n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized {
		err := n.ForwardGPUOptimized(inputs)
		if err != nil {
			if n.Debug {
				fmt.Printf("Optimized GPU forward failed, falling back to CPU: %v\n", err)
			}
			// Fall back to CPU
			n.forwardCPU(inputs)
		}
		return
	}

	// Fallback to existing implementation
	n.forwardCPU(inputs)
}

// Clean up GPU resources
func (n *Network[T]) CleanupOptimizedGPU() {
	if n.gpu.optimized == nil {
		return
	}

	for i, layer := range n.gpu.optimized.layers {
		if layer.inputBuffer != nil {
			layer.inputBuffer.Destroy()
		}
		if layer.outputBuffer != nil {
			layer.outputBuffer.Destroy()
		}
		if layer.weightBuffer != nil {
			layer.weightBuffer.Destroy()
		}
		if layer.biasBuffer != nil {
			layer.biasBuffer.Destroy()
		}
		if layer.stagingBuffer != nil {
			layer.stagingBuffer.Destroy()
		}
		if n.Debug {
			fmt.Printf("Cleaned up GPU resources for layer %d\n", i)
		}
	}

	n.gpu.optimized.layers = nil
	n.gpu.optimized.initialized = false
	n.gpu.optimized = nil
}
