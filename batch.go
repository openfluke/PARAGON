package paragon

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// Complete batch shader generation function
func batchShaderWGSL(batchSize int, layerSizes []struct{ in, out, act int }) string {
	// Find max layer size for temp array
	maxOutSize := 0
	for _, size := range layerSizes {
		if size.out > maxOutSize {
			maxOutSize = size.out
		}
	}

	// Start building shader
	shader := fmt.Sprintf(`
		struct Constants {
			batchSize: u32,
		}
		@group(0) @binding(0) var<uniform> constants: Constants;
		@group(0) @binding(1) var<storage, read> inBuf : array<f32>; // batchSize * inputSize
	`)

	// Add buffer bindings for each layer
	bindingIdx := 2
	for i := range layerSizes {
		shader += fmt.Sprintf(`
			@group(0) @binding(%d) var<storage, read_write> outBuf%d : array<f32>; // batchSize * layerSize
			@group(0) @binding(%d) var<storage, read> wBuf%d : array<f32>; // weights
			@group(0) @binding(%d) var<storage, read> bBuf%d : array<f32>; // biases
		`, bindingIdx, i, bindingIdx+1, i, bindingIdx+2, i)
		bindingIdx += 3
	}

	// Add activation function
	shader += `
		fn activate(x: f32, act: u32) -> f32 {
			if (act == 1u) { return select(0.0, x, x > 0.0); } // relu
			if (act == 2u) { return select(0.01 * x, x, x > 0.0); } // leaky_relu
			if (act == 3u) { return 1.0 / (1.0 + exp(-x)); } // sigmoid
			if (act == 4u) { return tanh(x); } // tanh
			return x;
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let globalIdx = gid.x;
			let batchSize = constants.batchSize;
	`

	// Process each layer
	for l, size := range layerSizes {
		inSize, outSize, act := size.in, size.out, size.act

		// Determine input source
		inputSource := "inBuf"
		inputLayerSize := layerSizes[0].in
		if l > 0 {
			inputSource = fmt.Sprintf("outBuf%d", l-1)
			inputLayerSize = layerSizes[l-1].out
		}

		shader += fmt.Sprintf(`
			// Layer %d: %d -> %d
			let layer%d_size = %du;
			let layer%d_input_size = %du;
			
			// Each thread processes one neuron for one sample
			let neuronIdx = globalIdx %% layer%d_size;
			let batchIdx = globalIdx / layer%d_size;
			
			if (batchIdx < batchSize && neuronIdx < layer%d_size) {
				var sum: f32 = bBuf%d[neuronIdx];
				
				// Compute weighted sum
				for (var i = 0u; i < layer%d_input_size; i++) {
					let inputIdx = batchIdx * layer%d_input_size + i;
					let weightIdx = neuronIdx * layer%d_input_size + i;
					sum += wBuf%d[weightIdx] * %s[inputIdx];
				}
				
				// Apply activation and store
				let activated = activate(sum, %du);
				let outputIdx = batchIdx * layer%d_size + neuronIdx;
				outBuf%d[outputIdx] = activated;
			}
		`, l, inSize, outSize,
			l, outSize,
			l, inputLayerSize,
			l, l,
			l, l,
			l, inputLayerSize,
			l, inputLayerSize,
			l, inputSource,
			act,
			l, outSize,
			l)
	}

	shader += `
		}
	`

	return shader
}

// Batch forward pass implementation
func (n *Network[T]) ForwardBatch(inputs [][][]float64) ([][]float64, error) {
	batchSize := len(inputs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	// Validate input dimensions
	in := n.Layers[n.InputLayer]
	for i, input := range inputs {
		if len(input) != in.Height || len(input[0]) != in.Width {
			return nil, fmt.Errorf("input %d dimension mismatch: want %dx%d, got %dx%d",
				i, in.Height, in.Width, len(input), len(input[0]))
		}
	}

	// CPU path: process sequentially
	if !n.WebGPUNative || any(*new(T)).(T) != T(float32(0)) {
		outputs := make([][]float64, batchSize)
		for i, input := range inputs {
			n.Forward(input)
			outputs[i] = n.GetOutput()
		}
		return outputs, nil
	}

	// GPU path: process batch in parallel
	return n.forwardGPUBatch(inputs)
}

// GPU batch forward implementation
func (n *Network[float32]) forwardGPUBatch(inputs [][][]float64) ([][]float64, error) {
	ensureGPU()
	batchSize := len(inputs)

	// Prepare batch input buffer
	inputSize := n.Layers[0].Width * n.Layers[0].Height
	flatInputs := make([]float32, batchSize*inputSize)

	for b, input := range inputs {
		offset := b * inputSize
		idx := 0
		for y := 0; y < len(input); y++ {
			for x := 0; x < len(input[y]); x++ {
				flatInputs[offset+idx] = float32(input[y][x])
				idx++
			}
		}
	}

	// Create batch-specific GPU resources
	if err := n.buildBatchGPUKernels(batchSize); err != nil {
		return nil, fmt.Errorf("failed to build batch kernels: %v", err)
	}

	// Upload input data
	ctx.queue.WriteBuffer(n.gpu.batchInBuf, 0, wgpu.ToBytes(flatInputs))

	// Upload batch size constant
	batchSizeData := []uint32{uint32(batchSize)}
	ctx.queue.WriteBuffer(n.gpu.batchConstBuf, 0, wgpu.ToBytes(batchSizeData))

	// Create command encoder
	enc, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Begin compute pass
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(n.gpu.batchPipeline)
	pass.SetBindGroup(0, n.gpu.batchBindGroup, nil)

	// Calculate workgroups
	// Need to process batchSize * neurons for each layer
	maxNeurons := 0
	for i := 1; i <= n.OutputLayer; i++ {
		neurons := n.Layers[i].Width * n.Layers[i].Height
		if neurons > maxNeurons {
			maxNeurons = neurons
		}
	}
	totalWork := uint32(batchSize * maxNeurons)
	workgroups := (totalWork + 255) / 256

	pass.DispatchWorkgroups(workgroups, 1, 1)
	pass.End()

	// Copy final output to staging buffer
	outputSize := n.Layers[n.OutputLayer].Width * n.Layers[n.OutputLayer].Height
	enc.CopyBufferToBuffer(
		n.gpu.batchOutBufs[n.OutputLayer-1], 0,
		n.gpu.batchStgBuf, 0,
		uint64(batchSize*outputSize*4))

	// Submit commands
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to finish command encoder: %v", err)
	}
	ctx.queue.Submit(cmd)

	// Wait for completion
	ctx.device.Poll(true, nil)

	// Map and read results
	done := make(chan wgpu.BufferMapAsyncStatus)
	err = n.gpu.batchStgBuf.MapAsync(wgpu.MapModeRead, 0, n.gpu.batchStgBuf.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Wait for mapping
	for {
		ctx.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("buffer mapping failed: %v", status)
			}
			goto readResults
		default:
			// Continue polling
		}
	}

readResults:
	// Read results
	data := n.gpu.batchStgBuf.GetMappedRange(0, 0)
	if data == nil {
		n.gpu.batchStgBuf.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	results := wgpu.FromBytes[float32](data)
	n.gpu.batchStgBuf.Unmap()

	// Convert results to output format
	outputs := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		outputs[b] = make([]float64, outputSize)
		for i := 0; i < outputSize; i++ {
			outputs[b][i] = float64(results[b*outputSize+i])
		}

		// Apply softmax if needed
		if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
			outputs[b] = Softmax(outputs[b])
		}
	}

	return outputs, nil
}

// Build GPU resources for batch processing
func (n *Network[float32]) buildBatchGPUKernels(batchSize int) error {
	ensureGPU()

	// Prepare layer sizes
	layerSizes := make([]struct{ in, out, act int }, n.OutputLayer)
	for l := 1; l <= n.OutputLayer; l++ {
		prev, cur := n.Layers[l-1], n.Layers[l]
		layerSizes[l-1] = struct{ in, out, act int }{
			prev.Width * prev.Height,
			cur.Width * cur.Height,
			actCodeOf(cur.Neurons[0][0].Activation),
		}
	}

	// Create buffers
	inputSize := n.Layers[0].Width * n.Layers[0].Height
	var err error

	// Constants buffer
	n.gpu.batchConstBuf, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  16, // Align to 16 bytes for uniform buffer
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create constants buffer: %v", err)
	}

	// Input buffer
	n.gpu.batchInBuf, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(batchSize * inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create batch input buffer: %v", err)
	}

	// Output buffers for each layer
	n.gpu.batchOutBufs = make([]*wgpu.Buffer, n.OutputLayer)
	for l := 1; l <= n.OutputLayer; l++ {
		size := n.Layers[l].Width * n.Layers[l].Height
		n.gpu.batchOutBufs[l-1], err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
			Size:  uint64(batchSize * size * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return fmt.Errorf("failed to create output buffer %d: %v", l-1, err)
		}
	}

	// Staging buffer for final output
	outputSize := n.Layers[n.OutputLayer].Width * n.Layers[n.OutputLayer].Height
	n.gpu.batchStgBuf, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(batchSize * outputSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create staging buffer: %v", err)
	}

	// Generate and compile shader
	shaderCode := batchShaderWGSL(batchSize, layerSizes)
	shaderModule, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return fmt.Errorf("failed to create shader module: %v", err)
	}

	// Create compute pipeline
	n.gpu.batchPipeline, err = ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shaderModule,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create compute pipeline: %v", err)
	}

	// Create bind group
	layout := n.gpu.batchPipeline.GetBindGroupLayout(0)
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: n.gpu.batchConstBuf, Size: n.gpu.batchConstBuf.GetSize()},
		{Binding: 1, Buffer: n.gpu.batchInBuf, Size: n.gpu.batchInBuf.GetSize()},
	}

	bindingIdx := uint32(2)
	for i := 0; i < n.OutputLayer; i++ {
		entries = append(entries,
			wgpu.BindGroupEntry{Binding: bindingIdx, Buffer: n.gpu.batchOutBufs[i], Size: n.gpu.batchOutBufs[i].GetSize()},
			wgpu.BindGroupEntry{Binding: bindingIdx + 1, Buffer: n.gpu.wBufs[i], Size: n.gpu.wBufs[i].GetSize()},
			wgpu.BindGroupEntry{Binding: bindingIdx + 2, Buffer: n.gpu.bBufs[i], Size: n.gpu.bBufs[i].GetSize()},
		)
		bindingIdx += 3
	}

	n.gpu.batchBindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  layout,
		Entries: entries,
	})
	if err != nil {
		return fmt.Errorf("failed to create bind group: %v", err)
	}

	return nil
}
