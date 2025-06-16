// webgpu.go
package paragon

import (
	"fmt"
	"sync"

	"github.com/openfluke/webgpu/wgpu"
)

type gpuContext struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	once     sync.Once
}

var ctx gpuContext

func ensureGPU() {
	ctx.once.Do(func() {
		ctx.instance = wgpu.CreateInstance(nil)
		var err error
		ctx.adapter, err = ctx.instance.RequestAdapter(&wgpu.RequestAdapterOptions{})
		if err != nil {
			panic(err)
		}
		ctx.device, err = ctx.adapter.RequestDevice(nil)
		if err != nil {
			panic(err)
		}
		ctx.queue = ctx.device.GetQueue()
	})
}

func newFloatBuf(data []float32, usage wgpu.BufferUsage) *wgpu.Buffer {
	ensureGPU()
	buf, err := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes(data),
		Usage:    usage,
	})
	if err != nil {
		panic(err)
	}
	return buf
}

func actCodeOf(name string) int {
	switch name {
	case "relu":
		return 1
	case "leaky_relu":
		return 2
	case "sigmoid":
		return 3
	case "tanh":
		return 4
	case "softmax": // Not used in shader, handled on CPU
		return 0
	default:
		return 0
	}
}

func allLayersShaderWGSL(layerSizes []struct{ in, out, act int }) string {
	maxOutSize := 0
	for _, size := range layerSizes {
		if size.out > maxOutSize {
			maxOutSize = size.out
		}
	}

	shader := `
		@group(0) @binding(0) var<storage, read> inBuf : array<f32>;
	`
	for i := range layerSizes {
		shader += fmt.Sprintf(`
			@group(0) @binding(%d) var<storage, read_write> outBuf%d : array<f32>;
			@group(0) @binding(%d) var<storage, read> wBuf%d : array<f32>;
			@group(0) @binding(%d) var<storage, read> bBuf%d : array<f32>;
		`, 1+i*3, i, 2+i*3, i, 3+i*3, i)
	}

	shader += fmt.Sprintf(`
		fn activate(x: f32, act: u32) -> f32 {
			if (act == 1u) { return select(0.0, x, x > 0.0); }
			if (act == 2u) { return select(0.01 * x, x, x > 0.0); }
			if (act == 3u) { return 1.0 / (1.0 + exp(-x)); }
			if (act == 4u) { return tanh(x); }
			return x;
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			var temp: array<f32, %d>;
			for (var i: u32 = 0u; i < %du; i++) {
				temp[i] = 0.0;
			}
	`, maxOutSize, maxOutSize)

	for l, size := range layerSizes {
		inSize, outSize, act := size.in, size.out, size.act
		inputSource := "inBuf"
		if l > 0 {
			inputSource = fmt.Sprintf("outBuf%d", l-1)
		}
		shader += fmt.Sprintf(`
			// Layer %d
			if (idx < %du) {
				var sum: f32 = bBuf%d[idx];
				for (var i: u32 = 0u; i < %du; i++) {
					sum += wBuf%d[idx * %du + i] * %s[i];
				}
				var activated = activate(sum, %du);
				temp[idx] = activated;
				outBuf%d[idx] = activated;
			}
		`, l, outSize, l, inSize, l, inSize, inputSource, act, l)
	}

	shader += `}`

	fmt.Printf("Generated Shader:\n%s\n", shader)
	return shader
}

func (n *Network[T]) BuildGPUKernels() {
	if !n.WebGPUNative || any(*new(T)).(T) != T(float32(0)) || len(n.Layers) < 2 {
		return
	}
	ensureGPU()

	// Clear existing GPU resources
	n.gpu.inBuf, n.gpu.stgBuf = nil, nil
	n.gpu.wBufs, n.gpu.bBufs, n.gpu.oBufs = nil, nil, nil
	n.gpu.pipel, n.gpu.layout, n.gpu.binds = nil, nil, nil
	n.gpu.stgBufs = nil // Clear staging buffers

	inElems := n.Layers[0].Width * n.Layers[0].Height
	if inElems <= 0 {
		panic(fmt.Sprintf("Invalid input layer size: %dx%d", n.Layers[0].Width, n.Layers[0].Height))
	}

	var err error
	n.gpu.inBuf, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(inElems) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		panic(fmt.Sprintf("Failed to create input buffer: %v", err))
	}

	layerSizes := make([]struct{ in, out, act int }, n.OutputLayer)
	n.gpu.oBufs = make([]*wgpu.Buffer, n.OutputLayer)
	n.gpu.stgBufs = make([]*wgpu.Buffer, n.OutputLayer) // ALLOCATE staging buffer slice

	for l := 1; l <= n.OutputLayer; l++ {
		prev, cur := n.Layers[l-1], n.Layers[l]
		in := prev.Width * prev.Height
		out := cur.Width * cur.Height
		if in <= 0 || out <= 0 {
			panic(fmt.Sprintf("Invalid layer %d size: in=%d, out=%d", l, in, out))
		}
		layerSizes[l-1] = struct{ in, out, act int }{in, out, actCodeOf(cur.Neurons[0][0].Activation)}

		// Prepare weights and biases
		w := make([]float32, in*out)
		b := make([]float32, out)
		idx := 0
		for y := 0; y < cur.Height; y++ {
			for x := 0; x < cur.Width; x++ {
				for _, c := range cur.Neurons[y][x].Inputs {
					w[idx] = float32(any(c.Weight).(T))
					idx++
				}
				b[y*cur.Width+x] = float32(any(cur.Neurons[y][x].Bias).(T))
			}
		}
		n.gpu.wBufs = append(n.gpu.wBufs, newFloatBuf(w, wgpu.BufferUsageStorage))
		n.gpu.bBufs = append(n.gpu.bBufs, newFloatBuf(b, wgpu.BufferUsageStorage))

		// Create output buffer
		n.gpu.oBufs[l-1], err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
			Size:  uint64(out) * 4,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			panic(fmt.Sprintf("Failed to create output buffer for layer %d: %v", l-1, err))
		}

		// Create staging buffer (FIX: this was creating oBufs twice)
		n.gpu.stgBufs[l-1], err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
			Size:  uint64(out) * 4,
			Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			panic(fmt.Sprintf("Failed to create staging buffer for layer %d: %v", l-1, err))
		}
	}

	// Generate shader and create pipeline
	code := allLayersShaderWGSL(layerSizes)
	mod, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		panic(fmt.Sprintf("Failed to create shader module: %v", err))
	}

	pipe, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	if err != nil {
		panic(fmt.Sprintf("Failed to create compute pipeline: %v", err))
	}
	layout := pipe.GetBindGroupLayout(0)

	// Create bind group
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: n.gpu.inBuf, Size: n.gpu.inBuf.GetSize()},
	}
	for i, oBuf := range n.gpu.oBufs {
		entries = append(entries,
			wgpu.BindGroupEntry{Binding: uint32(1 + i*3), Buffer: oBuf, Size: oBuf.GetSize()},
			wgpu.BindGroupEntry{Binding: uint32(2 + i*3), Buffer: n.gpu.wBufs[i], Size: n.gpu.wBufs[i].GetSize()},
			wgpu.BindGroupEntry{Binding: uint32(3 + i*3), Buffer: n.gpu.bBufs[i], Size: n.gpu.bBufs[i].GetSize()},
		)
	}
	bind, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  layout,
		Entries: entries,
	})
	if err != nil {
		panic(fmt.Sprintf("Failed to create bind group: %v", err))
	}

	n.gpu.pipel = []*wgpu.ComputePipeline{pipe}
	n.gpu.layout = []*wgpu.BindGroupLayout{layout}
	n.gpu.binds = []*wgpu.BindGroup{bind}

	if n.Debug {
		fmt.Printf("GPU kernels built successfully with %d layers\n", n.OutputLayer)
	}
}

func (n *Network[float32]) forwardGPU(sample [][]float64) (*wgpu.Buffer, error) {
	if n.Debug {
		fmt.Println("Starting GPU forward pass...")
	}

	ensureGPU()

	// Flatten input
	inFlat := make([]float32, 0, len(sample)*len(sample[0]))
	for _, row := range sample {
		for _, v := range row {
			inFlat = append(inFlat, float32(v))
		}
	}

	// Verify GPU state
	if n.gpu.inBuf == nil || len(n.gpu.pipel) == 0 || len(n.gpu.binds) == 0 {
		return nil, fmt.Errorf("GPU not properly initialized")
	}

	// Write input data
	ctx.queue.WriteBuffer(n.gpu.inBuf, 0, wgpu.ToBytes(inFlat))

	// Create command encoder
	enc, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Begin compute pass
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(n.gpu.pipel[0])
	pass.SetBindGroup(0, n.gpu.binds[0], nil)

	// Calculate workgroups for the largest layer output
	maxElems := uint32(0)
	for i := 1; i <= n.OutputLayer; i++ {
		elems := uint32(n.Layers[i].Width * n.Layers[i].Height)
		if elems > maxElems {
			maxElems = elems
		}
	}
	workgroups := (maxElems + 255) / 256
	if n.Debug {
		fmt.Printf("Dispatching %d workgroups for %d elements\n", workgroups, maxElems)
	}
	pass.DispatchWorkgroups(workgroups, 1, 1)
	pass.End()

	// Copy all output buffers to staging buffers
	for i, oBuf := range n.gpu.oBufs {
		if n.gpu.stgBufs[i] == nil {
			return nil, fmt.Errorf("staging buffer %d is nil", i)
		}
		enc.CopyBufferToBuffer(oBuf, 0, n.gpu.stgBufs[i], 0, oBuf.GetSize())
	}

	// Submit commands
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to finish command encoder: %v", err)
	}
	ctx.queue.Submit(cmd)

	if n.Debug {
		fmt.Println("GPU commands submitted successfully")
	}

	// Return the last staging buffer
	return n.gpu.stgBufs[len(n.gpu.stgBufs)-1], nil
}

// New error-safe GPU forward pass method
func (n *Network[T]) forwardGPUWithErrorHandling(inputs [][]float64, replayed map[int]int) error {
	// Validate GPU state
	if err := n.validateGPUState(); err != nil {
		return fmt.Errorf("GPU state validation failed: %v", err)
	}

	// Execute main GPU computation
	if err := n.executeGPUComputation(inputs); err != nil {
		return fmt.Errorf("GPU computation failed: %v", err)
	}

	// Read results with proper synchronization
	gpuResults, err := n.readGPUResultsSafe()
	if err != nil {
		return fmt.Errorf("failed to read GPU results: %v", err)
	}

	// Apply results to network layers
	if err := n.applyGPUResults(gpuResults); err != nil {
		return fmt.Errorf("failed to apply GPU results: %v", err)
	}

	// Handle replay logic (simplified for GPU)
	n.handleGPUReplay(gpuResults, replayed)

	// Final softmax pass
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}

	return nil
}

// Validate GPU buffers and state before computation
func (n *Network[T]) validateGPUState() error {
	// Check input buffer
	if n.gpu.inBuf == nil {
		return fmt.Errorf("input buffer not initialized")
	}

	// Check staging buffers exist (don't check mapped state)
	if len(n.gpu.stgBufs) != n.OutputLayer {
		return fmt.Errorf("staging buffer count mismatch: got %d, expected %d",
			len(n.gpu.stgBufs), n.OutputLayer)
	}

	// Only check that buffers are not nil
	for i, buf := range n.gpu.stgBufs {
		if buf == nil {
			return fmt.Errorf("staging buffer %d is nil", i)
		}
		// DO NOT call GetMappedRange here!
	}

	// Check output buffers
	if len(n.gpu.oBufs) != n.OutputLayer {
		return fmt.Errorf("output buffer count mismatch: got %d, expected %d",
			len(n.gpu.oBufs), n.OutputLayer)
	}

	for i, buf := range n.gpu.oBufs {
		if buf == nil {
			return fmt.Errorf("output buffer %d is nil", i)
		}
	}

	// Check weight and bias buffers
	if len(n.gpu.wBufs) != n.OutputLayer {
		return fmt.Errorf("weight buffer count mismatch: got %d, expected %d",
			len(n.gpu.wBufs), n.OutputLayer)
	}

	if len(n.gpu.bBufs) != n.OutputLayer {
		return fmt.Errorf("bias buffer count mismatch: got %d, expected %d",
			len(n.gpu.bBufs), n.OutputLayer)
	}

	// Check pipeline and bind group
	if len(n.gpu.pipel) == 0 || n.gpu.pipel[0] == nil {
		return fmt.Errorf("compute pipeline not initialized")
	}

	if len(n.gpu.binds) == 0 || n.gpu.binds[0] == nil {
		return fmt.Errorf("bind group not initialized")
	}

	return nil
}

// Execute the GPU computation
func (n *Network[T]) executeGPUComputation(inputs [][]float64) error {
	// Type assertion to check if we can call forwardGPU
	if netFloat32, ok := any(n).(*Network[float32]); ok {
		finalBuf, err := netFloat32.forwardGPU(inputs)
		if err != nil {
			return fmt.Errorf("GPU forward pass failed: %v", err)
		}
		if finalBuf == nil {
			return fmt.Errorf("GPU computation returned nil buffer")
		}
		// Store the compute buffer for later use
		n.gpu.computeBuf = finalBuf
	} else {
		return fmt.Errorf("GPU computation only supported for float32 networks")
	}

	// Wait for GPU operations to complete
	ctx.device.Poll(true, nil)

	return nil
}

func (n *Network[T]) TestGPUDirect(inputs [][]float64) {
	if !n.WebGPUNative {
		fmt.Println("GPU not enabled")
		return
	}

	fmt.Println("Testing GPU directly without validation...")

	// Try to run GPU forward pass directly
	if netFloat32, ok := any(n).(*Network[float32]); ok {
		result, err := netFloat32.forwardGPU(inputs)
		if err != nil {
			fmt.Printf("GPU forward failed: %v\n", err)
			return
		}

		fmt.Printf("GPU forward succeeded, result buffer: %v\n", result != nil)

		// Try to read results
		ctx.device.Poll(true, nil)

		// Map and read the final buffer
		done := make(chan bool)
		var mapErr error

		err = result.MapAsync(wgpu.MapModeRead, 0, result.GetSize(),
			func(status wgpu.BufferMapAsyncStatus) {
				if status == wgpu.BufferMapAsyncStatusSuccess {
					fmt.Println("Buffer mapped successfully")
				} else {
					mapErr = fmt.Errorf("map failed with status: %v", status)
				}
				done <- true
			})

		if err != nil {
			fmt.Printf("MapAsync failed: %v\n", err)
			return
		}

		// Wait for mapping
		<-done

		if mapErr != nil {
			fmt.Printf("Mapping error: %v\n", mapErr)
			return
		}

		// Read data
		data := result.GetMappedRange(0, 0)
		if data == nil {
			fmt.Println("GetMappedRange returned nil")
			return
		}

		values := wgpu.FromBytes[float32](data)
		fmt.Printf("GPU output: %v\n", values)

		result.Unmap()
	}
}

// Safely read GPU results with proper synchronization
func (n *Network[T]) readGPUResultsSafe() ([][]float32, error) {
	gpuResults := make([][]float32, n.OutputLayer)

	// Process each layer's results sequentially
	for l := 0; l < n.OutputLayer; l++ {
		buf := n.gpu.stgBufs[l]

		// Map buffer with timeout and error handling
		mapped := make(chan wgpu.BufferMapAsyncStatus, 1)

		err := buf.MapAsync(wgpu.MapModeRead, 0, buf.GetSize(),
			func(status wgpu.BufferMapAsyncStatus) {
				mapped <- status
			})

		if err != nil {
			return nil, fmt.Errorf("failed to initiate buffer mapping for layer %d: %v", l, err)
		}

		// Wait for mapping to complete with timeout
		timeout := 0
		for {
			ctx.device.Poll(true, nil)
			select {
			case status := <-mapped:
				if status != wgpu.BufferMapAsyncStatusSuccess {
					return nil, fmt.Errorf("buffer mapping failed for layer %d: %v", l, status)
				}
				goto readBuffer
			default:
				timeout++
				if timeout > 10000 { // Prevent infinite loop
					return nil, fmt.Errorf("buffer mapping timeout for layer %d", l)
				}
			}
		}

	readBuffer:
		// Read the buffer data
		raw := buf.GetMappedRange(0, uint(buf.GetSize()))
		if raw == nil {
			buf.Unmap()
			return nil, fmt.Errorf("failed to get mapped range for layer %d", l)
		}

		gpuResults[l] = wgpu.FromBytes[float32](raw)
		buf.Unmap()

		if n.Debug {
			fmt.Printf("Successfully read %d elements from GPU layer %d\n", len(gpuResults[l]), l)
		}
	}

	return gpuResults, nil
}

// Apply GPU results to network layers
func (n *Network[T]) applyGPUResults(gpuResults [][]float32) error {
	if len(gpuResults) != n.OutputLayer {
		return fmt.Errorf("GPU results length mismatch: got %d, expected %d", len(gpuResults), n.OutputLayer)
	}

	// Apply results to all layers computed by GPU (1 to OutputLayer)
	for l := 1; l <= n.OutputLayer; l++ {
		g := &n.Layers[l]
		out := gpuResults[l-1] // gpuResults[0] is layer 1, gpuResults[1] is layer 2, etc.
		expectedElems := g.Width * g.Height

		if len(out) != expectedElems {
			return fmt.Errorf("GPU output size mismatch for layer %d: got %d, expected %d", l, len(out), expectedElems)
		}

		idx := 0
		for y := 0; y < g.Height; y++ {
			for x := 0; x < g.Width; x++ {
				g.Neurons[y][x].Value = T(out[idx])
				idx++
			}
		}
	}

	return nil
}

// Handle replay logic for GPU computation
func (n *Network[T]) handleGPUReplay(gpuResults [][]float32, replayed map[int]int) {
	// Simplified replay logic for GPU - cache the results
	for l := 1; l < n.OutputLayer; l++ {
		layer := &n.Layers[l]

		// Cache GPU results for replay layers
		if l-1 < len(gpuResults) && len(gpuResults[l-1]) > 0 {
			outputValues := make([]float64, len(gpuResults[l-1]))
			for j, v := range gpuResults[l-1] {
				outputValues[j] = float64(v)
			}
			layer.CachedOutputs = CastFloat64SliceToT[T](outputValues)
		}

		// Handle dynamic replay if enabled
		if layer.ReplayEnabled && layer.ReplayGateFunc != nil {
			score := layer.ReplayGateFunc(nil)
			nreps := layer.ReplayBudget
			if layer.ReplayGateToReps != nil {
				nreps = layer.ReplayGateToReps(score)
			}
			if nreps > layer.ReplayBudget {
				nreps = layer.ReplayBudget
			}

			// For GPU, we just record the replay count
			replayed[l] += nreps

			if n.Debug {
				fmt.Printf("Layer %d: replay score=%.3f, reps=%d\n", l, score, nreps)
			}
		}
	}
}

func (n *Network[T]) VerifyGPUSetup() error {
	if !n.WebGPUNative {
		return fmt.Errorf("WebGPU not enabled")
	}

	if n.gpu.inBuf == nil {
		return fmt.Errorf("input buffer not initialized")
	}

	if len(n.gpu.stgBufs) == 0 {
		return fmt.Errorf("no staging buffers created")
	}

	for i, buf := range n.gpu.stgBufs {
		if buf == nil {
			return fmt.Errorf("staging buffer %d is nil", i)
		}
	}

	if len(n.gpu.pipel) == 0 || n.gpu.pipel[0] == nil {
		return fmt.Errorf("compute pipeline not created")
	}

	if len(n.gpu.binds) == 0 || n.gpu.binds[0] == nil {
		return fmt.Errorf("bind group not created")
	}

	return nil
}
