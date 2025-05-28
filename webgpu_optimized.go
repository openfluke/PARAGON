// webgpu_optimized.go
package paragon

import (
	"fmt"
	"math"

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

// Initialize both forward and backward GPU compute
func (n *Network[T]) InitializeGPUComplete() error {
	// Initialize forward pass
	if err := n.InitializeOptimizedGPU(); err != nil {
		return fmt.Errorf("failed to initialize GPU forward pass: %v", err)
	}

	// Initialize backward pass
	if err := n.InitializeGPUBackward(); err != nil {
		return fmt.Errorf("failed to initialize GPU backward pass: %v", err)
	}

	return nil
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

// ---------------------------------------------------------------------------
//
//	createLayerCompute  – forward-pass pipeline for one layer
//	Shares weight & bias buffers with the GPU-backward engine.
//
// ---------------------------------------------------------------------------
func (n *Network[T]) createLayerCompute(layerIdx int) (*GPULayerCompute, error) {

	prev := n.Layers[layerIdx-1]
	curr := n.Layers[layerIdx]

	inSize := uint32(prev.Width * prev.Height)
	outSize := uint32(curr.Width * curr.Height)
	if inSize == 0 || outSize == 0 {
		return nil, fmt.Errorf("invalid dims: in=%d out=%d", inSize, outSize)
	}

	// ── WGSL shader -------------------------------------------------------
	shader := n.generateLayerShader(layerIdx, inSize, outSize)
	mod, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("L%d_shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return nil, err
	}
	defer mod.Release()

	// ── Bind-group layout -------------------------------------------------
	bgl, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("L%d_bgl", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingType_ReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingType_Storage}},
			{Binding: 2, Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingType_ReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStage_Compute,
				Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingType_ReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, err
	}

	pl, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("L%d_pl", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		bgl.Release()
		return nil, err
	}
	defer pl.Release()

	cp, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("L%d_cp", layerIdx),
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module: mod, EntryPoint: "main",
		},
	})
	if err != nil {
		bgl.Release()
		return nil, err
	}

	lc := &GPULayerCompute{
		pipeline:        cp,
		bindGroupLayout: bgl,
		inputSize:       inSize,
		outputSize:      outSize,
		layerIndex:      layerIdx,
	}

	// ── own input / output / staging buffers ------------------------------
	lc.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("L%d_in", layerIdx),
		Size:  uint64(inSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		lc.cleanup()
		return nil, err
	}

	lc.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("L%d_out", layerIdx),
		Size:  uint64(outSize) * 4,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		lc.cleanup()
		return nil, err
	}

	lc.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("L%d_stage", layerIdx),
		Size:  uint64(outSize) * 4,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		lc.cleanup()
		return nil, err
	}

	// ── weight / bias – reuse from backward if present --------------------
	if n.gpu.backward != nil && layerIdx-1 < len(n.gpu.backward.layers) {
		lc.weightBuffer = n.gpu.backward.layers[layerIdx-1].weightBuffer
		lc.biasBuffer = n.gpu.backward.layers[layerIdx-1].biasBuffer
	} else {
		// Fallback for CPU-only builds
		w, b := n.extractLayerWeightsAndBiases(layerIdx)

		lc.weightBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    fmt.Sprintf("L%d_W", layerIdx),
			Contents: wgpu.ToBytes(w),
			Usage:    wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
		})
		if err != nil {
			lc.cleanup()
			return nil, err
		}

		lc.biasBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    fmt.Sprintf("L%d_B", layerIdx),
			Contents: wgpu.ToBytes(b),
			Usage:    wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
		})
		if err != nil {
			lc.cleanup()
			return nil, err
		}
	}

	// ── bind group --------------------------------------------------------
	lc.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("L%d_bg", layerIdx),
		Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: lc.inputBuffer, Size: lc.inputBuffer.GetSize()},
			{Binding: 1, Buffer: lc.outputBuffer, Size: lc.outputBuffer.GetSize()},
			{Binding: 2, Buffer: lc.weightBuffer, Size: lc.weightBuffer.GetSize()},
			{Binding: 3, Buffer: lc.biasBuffer, Size: lc.biasBuffer.GetSize()},
		},
	})
	if err != nil {
		lc.cleanup()
		return nil, err
	}

	// ── work-group sizing -------------------------------------------------
	lc.workgroupsX = (outSize + 255) / 256
	lc.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created L%d  %dx%d→%dx%d  wg:%d\n",
			layerIdx, prev.Width, prev.Height, curr.Width, curr.Height, lc.workgroupsX)
	}
	return lc, nil
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

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input: array<f32>;
		@group(0) @binding(1) var<storage, read_write> output: array<f32>;
		@group(0) @binding(2) var<storage, read> weights: array<f32>;
		@group(0) @binding(3) var<storage, read> biases: array<f32>;

		%s

		@compute @workgroup_size(64, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let output_idx = global_id.x;
			
			if (output_idx >= %du) {
				return;
			}

			// Initialize with bias
			var sum: f32 = biases[output_idx];
			
			// Compute weighted sum
			let weight_offset = output_idx * %du;
			for (var input_idx: u32 = 0u; input_idx < %du; input_idx = input_idx + 1u) {
				let weight_idx = weight_offset + input_idx;
				sum = sum + weights[weight_idx] * input[input_idx];
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

	// Initialize weight matrix and bias array
	weights := make([]float32, outputSize*inputSize)
	biases := make([]float32, outputSize)

	if n.Debug {
		fmt.Printf("Extracting weights for layer %d: %d inputs -> %d outputs\n",
			layerIdx, inputSize, outputSize)
	}

	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			// Extract bias
			biases[neuronIdx] = float32(any(neuron.Bias).(T))

			// Initialize weight row to zero
			weightRowStart := neuronIdx * inputSize
			for i := 0; i < inputSize; i++ {
				weights[weightRowStart+i] = 0.0
			}

			// Warn if neuron has no inputs
			if len(neuron.Inputs) == 0 {
				if n.Debug {
					fmt.Printf("Warning: Layer %d, Neuron (%d,%d) has no inputs\n", layerIdx, x, y)
				}
			}

			// Map connections to weight matrix
			for _, conn := range neuron.Inputs {
				inputIdx := conn.SourceY*prevLayer.Width + conn.SourceX

				// Bounds check
				if inputIdx >= 0 && inputIdx < inputSize {
					weightIdx := weightRowStart + inputIdx
					if weightIdx < len(weights) {
						weights[weightIdx] = float32(any(conn.Weight).(T))
					} else {
						if n.Debug {
							fmt.Printf("Warning: Invalid weight index %d for layer %d\n", weightIdx, layerIdx)
						}
					}
				}
			}
		}
	}

	if n.Debug {
		// Count non-zero weights and check distribution
		nonZeroCount := 0
		minWeight := float32(math.MaxFloat32)
		maxWeight := float32(-math.MaxFloat32)
		for _, w := range weights {
			if w != 0.0 {
				nonZeroCount++
				if w < minWeight {
					minWeight = w
				}
				if w > maxWeight {
					maxWeight = w
				}
			}
		}
		fmt.Printf("Weight matrix: %d total, %d non-zero (%.1f%% sparse)\n",
			len(weights), nonZeroCount, float64(len(weights)-nonZeroCount)/float64(len(weights))*100)
		fmt.Printf("Weight range: [%.6f, %.6f]\n", minWeight, maxWeight)
	}

	return weights, biases
}

// Optimized forward pass using proper GPU parallelization
func (n *Network[T]) ForwardGPUOptimized(inputs [][]float64) error {
	if !n.gpu.optimized.initialized {
		return fmt.Errorf("optimized GPU not initialized")
	}

	// ── flatten input once ────────────────────────────────────────────────
	inputData := make([]float32, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, float32(val))
		}
	}

	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}

	for i, lc := range n.gpu.optimized.layers {

		// 👉  only the **first** layer receives a WriteBuffer;
		//     every later layer receives its data via the copy issued
		//     at the end of the previous iteration.
		if i == 0 {
			ctx.queue.WriteBuffer(lc.inputBuffer, 0, wgpu.ToBytes(inputData))
		}

		pass := encoder.BeginComputePass(nil)
		pass.SetPipeline(lc.pipeline)
		pass.SetBindGroup(0, lc.bindGroup, nil)
		pass.DispatchWorkgroups(lc.workgroupsX, lc.workgroupsY, 1)
		pass.End()

		if i < len(n.gpu.optimized.layers)-1 {
			next := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				lc.outputBuffer, 0,
				next.inputBuffer, 0,
				uint64(lc.outputSize)*4,
			)
		} else { // final layer → stage for CPU read-back
			encoder.CopyBufferToBuffer(
				lc.outputBuffer, 0,
				lc.stagingBuffer, 0,
				uint64(lc.outputSize)*4,
			)
		}
	}

	cmd, _ := encoder.Finish(nil)
	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	final := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	out, err := n.readStagingBuffer(final.stagingBuffer, int(final.outputSize))
	if err != nil {
		return err
	}

	n.applyFinalOutput(out)
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

	// Also clean up backward pass resources
	n.CleanupGPUBackward()
}

func (n *Network[float32]) syncForwardWeightsToGPU() {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return
	}
	for _, lc := range n.gpu.optimized.layers {
		w, b := n.extractLayerWeightsAndBiases(lc.layerIndex)
		ctx.queue.WriteBuffer(lc.weightBuffer, 0, wgpu.ToBytes(w))
		ctx.queue.WriteBuffer(lc.biasBuffer, 0, wgpu.ToBytes(b))
	}
}

func (n *Network[float32]) syncBackwardWeightsToGPU() {
	if n.gpu.backward == nil || !n.gpu.backward.initialized {
		return
	}
	for _, lb := range n.gpu.backward.layers {
		w, b := n.extractLayerWeightsAndBiases(lb.layerIndex)
		ctx.queue.WriteBuffer(lb.weightBuffer, 0, wgpu.ToBytes(w))
		ctx.queue.WriteBuffer(lb.biasBuffer, 0, wgpu.ToBytes(b))
	}
}
