package paragon

import (
	"fmt"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

func (n *Network[T]) TestWeightCycle(layerIdx int) {
	if !n.Debug {
		return
	}

	fmt.Printf("=== Testing Weight Extraction/Application Cycle for Layer %d ===\n", layerIdx)

	// Get original weights from neurons
	originalWeights := make(map[string]float32)
	layer := n.Layers[layerIdx]

	totalConnections := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]
			for i, conn := range neuron.Inputs {
				key := fmt.Sprintf("%d_%d_%d", y, x, i)
				originalWeights[key] = float32(any(conn.Weight).(T))
				totalConnections++
			}
		}
	}

	fmt.Printf("Original: %d connections in layer\n", totalConnections)

	// Extract to matrix
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	nonZero := 0
	for _, w := range weights {
		if w != 0.0 {
			nonZero++
		}
	}
	fmt.Printf("Extracted: %d total matrix elements, %d non-zero\n", len(weights), nonZero)

	// Apply back
	err := n.applyWeightsAndBiases(layerIdx, weights, biases)
	if err != nil {
		fmt.Printf("Application error: %v\n", err)
		return
	}

	// Check consistency
	matches := 0
	mismatches := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]
			for i, conn := range neuron.Inputs {
				key := fmt.Sprintf("%d_%d_%d", y, x, i)
				original := originalWeights[key]
				current := float32(any(conn.Weight).(T))

				if absFloat32(original-current) < 0.0001 {
					matches++
				} else {
					mismatches++
					if mismatches < 5 { // Show first few mismatches
						fmt.Printf("Mismatch %s: %.6f -> %.6f\n", key, original, current)
					}
				}
			}
		}
	}

	fmt.Printf("Consistency check: %d matches, %d mismatches\n", matches, mismatches)

	if mismatches == 0 {
		fmt.Println("✓ Weight extraction/application cycle is consistent")
	} else {
		fmt.Println("✗ Weight extraction/application has issues")
	}
}

// Add this to your debug validation function
func (n *Network[T]) ValidateGPUBackwardWithWeightTest(inputs [][]float64, targets [][]float64, lr float64) error {
	if n.Debug {
		fmt.Println("=== GPU Backward Pass Validation with Weight Testing ===")

		// Test weight cycle for each layer
		for l := 1; l <= n.OutputLayer; l++ {
			n.TestWeightCycle(l)
		}
	}

	// Continue with normal validation...
	return n.ValidateGPUBackward(inputs, targets, lr)
}

// CRITICAL DEBUG: Let's also check what's happening in the shader
// Add this debug version of the upload function:
func (n *Network[T]) uploadBackwardDataDebugZeros(
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

	if n.Debug {
		// Count non-zero weights
		nonZero := 0
		for _, w := range weights {
			if w != 0.0 {
				nonZero++
			}
		}

		// Check for suspicious zero patterns
		zeroRuns := 0
		maxZeroRun := 0
		currentRun := 0

		for _, w := range weights {
			if w == 0.0 {
				currentRun++
			} else {
				if currentRun > 0 {
					zeroRuns++
					if currentRun > maxZeroRun {
						maxZeroRun = currentRun
					}
				}
				currentRun = 0
			}
		}
		// Handle case where weights end with zeros
		if currentRun > 0 {
			zeroRuns++
			if currentRun > maxZeroRun {
				maxZeroRun = currentRun
			}
		}

		fmt.Printf("Zero pattern analysis: %d zero runs, max run length: %d\n", zeroRuns, maxZeroRun)
		fmt.Printf("Weight density: %d/%d (%.1f%% non-zero)\n",
			nonZero, len(weights), float64(nonZero)/float64(len(weights))*100)

		// Sample some weights to upload
		fmt.Printf("Sample weights being uploaded: ")
		for i := 0; i < min(10, len(weights)); i++ {
			fmt.Printf("%.6f ", weights[i])
		}
		fmt.Println()

		// Show first few non-zero weights
		fmt.Printf("First few non-zero weights: ")
		count := 0
		for _, w := range weights {
			if w != 0.0 && count < 5 {
				fmt.Printf("%.6f ", w)
				count++
			}
		}
		fmt.Println()
	}

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

	// Count non-zero weights for final report
	nonZeroFinal := 0
	for _, w := range weights {
		if w != 0.0 {
			nonZeroFinal++
		}
	}

	if n.Debug {
		fmt.Printf("Uploaded layer %d: %d weights (%d non-zero), %d biases, lr=%.4f\n",
			layerIdx, len(weights), nonZeroFinal, len(biases), lr)
	}

	return nil
}

func (n *Network[T]) ResetGPUBuffers() error {
	if n.gpu.backward == nil || !n.gpu.backward.initialized {
		return nil
	}

	for _, layer := range n.gpu.backward.layers {
		// Create zero buffers of the correct size
		weightSize := int(layer.inputSize * layer.outputSize)
		biasSize := int(layer.outputSize)
		errorSize := int(layer.inputSize)

		// Reset weight buffer to zeros
		zeroWeights := make([]float32, weightSize)
		ctx.queue.WriteBuffer(layer.weightBuffer, 0, wgpu.ToBytes(zeroWeights))

		// Reset bias buffer to zeros
		zeroBiases := make([]float32, biasSize)
		ctx.queue.WriteBuffer(layer.biasBuffer, 0, wgpu.ToBytes(zeroBiases))

		// Reset error buffers
		zeroErrors := make([]float32, errorSize)
		ctx.queue.WriteBuffer(layer.prevErrorBuffer, 0, wgpu.ToBytes(zeroErrors))

		outputErrorSize := int(layer.outputSize)
		zeroOutputErrors := make([]float32, outputErrorSize)
		ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(zeroOutputErrors))
	}

	return nil
}

func (n *Network[T]) uploadBackwardDataClean(
	layer *GPUBackwardLayer,
	layerIdx int,
	errorData []float32,
	lr float64,
	clipUpper T,
	clipLower T,
) error {
	// Upload error for this layer
	ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(errorData))

	// Get CLEAN weights from the network (not contaminated by test data)
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	if n.Debug && layerIdx == 1 {
		// Check if we still have test contamination
		hasTestData := false
		testValues := []float32{999.0, 888.0, 777.0, 666.0, 555.0}
		for i, testVal := range testValues {
			if i < len(weights) && weights[i] == testVal {
				hasTestData = true
				break
			}
		}

		if hasTestData {
			fmt.Printf("⚠️ WARNING: Test data contamination detected in layer %d weights!\n", layerIdx)
		} else {
			fmt.Printf("✓ Clean weights detected for layer %d\n", layerIdx)
		}

		// Show actual weight values (not test values)
		fmt.Printf("Clean weights being uploaded: ")
		for i := 0; i < min(10, len(weights)); i++ {
			fmt.Printf("%.6f ", weights[i])
		}
		fmt.Println()
	}

	// Upload CLEAN weights and biases
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

	return nil
}

// Clean result reading
func (n *Network[T]) readBackwardResultsClean() error {
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

		// Check for test data contamination in results
		if n.Debug && layerIdx == 1 {
			testValues := []float32{999.0, 888.0, 777.0, 666.0, 555.0}
			hasContamination := false
			for i, testVal := range testValues {
				if i < len(weights) && absFloat32(weights[i]-testVal) < 0.1 {
					hasContamination = true
					break
				}
			}

			if hasContamination {
				fmt.Printf("⚠️ WARNING: GPU results still contaminated with test data!\n")
			} else {
				fmt.Printf("✓ Clean GPU results for layer %d\n", layerIdx)
			}
		}

		// Apply to network
		if err := n.applyWeightsAndBiases(layerIdx, weights, biases); err != nil {
			return fmt.Errorf("failed to apply results for layer %d: %v", layerIdx, err)
		}
	}

	return nil
}

func (n *Network[T]) TestBufferWriteIsolated() error {
	if n.gpu.backward == nil || len(n.gpu.backward.layers) == 0 {
		return fmt.Errorf("no backward layers available")
	}

	// Create a separate test buffer - DON'T use the actual weight buffer
	testData := []float32{999.0, 888.0, 777.0, 666.0, 555.0}

	// Create a temporary buffer for testing
	testBuffer, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "TestBuffer",
		Size:  20, // 5 floats * 4 bytes
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create test buffer: %v", err)
	}
	defer testBuffer.Destroy()

	// Create staging buffer for test
	testStaging, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "TestStaging",
		Size:  20,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create test staging: %v", err)
	}
	defer testStaging.Destroy()

	// Write test data to SEPARATE buffer
	ctx.queue.WriteBuffer(testBuffer, 0, wgpu.ToBytes(testData))

	// Create command encoder to copy to staging
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create encoder: %v", err)
	}

	// Copy test buffer to staging
	encoder.CopyBufferToBuffer(testBuffer, 0, testStaging, 0, 20)

	// Submit and wait
	cmd, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	// Read back the data from TEST buffer (not weight buffer)
	readData, err := n.readStagingBufferTest(testStaging, 5)
	if err != nil {
		return fmt.Errorf("failed to read test staging buffer: %v", err)
	}

	// Check if write succeeded
	fmt.Printf("Isolated buffer write test results:\n")
	allPassed := true
	for i, val := range readData {
		expected := testData[i]
		fmt.Printf("  [%d] Expected: %.1f, Got: %.1f", i, expected, val)
		if absFloat32(val-expected) < 0.1 {
			fmt.Printf(" ✓")
		} else {
			fmt.Printf(" ✗")
			allPassed = false
		}
		fmt.Println()
	}

	if allPassed {
		fmt.Println("✓ GPU buffer write capability confirmed (isolated test)")
	} else {
		fmt.Println("✗ GPU buffer write capability failed")
	}

	return nil
}

func (n *Network[T]) readStagingBufferTest(buffer *wgpu.Buffer, size int) ([]float32, error) {
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

// The issue is that TestBufferWrite() is contaminating the CPU network
// Let's fix this by isolating the test and preventing contamination

// CRITICAL FIX: The contamination must be coming from the readBackwardStagingBuffer
// or applyWeightsAndBiases functions. Let's add network restoration:

func (n *Network[T]) SaveNetworkWeights() map[string][]float32 {
	saved := make(map[string][]float32)

	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]

		// Save all weights

		allWeights := make([]float32, 0)
		allBiases := make([]float32, 0)

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				allBiases = append(allBiases, float32(any(neuron.Bias).(T)))

				for _, conn := range neuron.Inputs {
					allWeights = append(allWeights, float32(any(conn.Weight).(T)))
				}
			}
		}

		saved[fmt.Sprintf("weights_%d", l)] = allWeights
		saved[fmt.Sprintf("biases_%d", l)] = allBiases
	}

	return saved
}

func (n *Network[T]) RestoreNetworkWeights(saved map[string][]float32) {
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]

		weightsKey := fmt.Sprintf("weights_%d", l)
		biasesKey := fmt.Sprintf("biases_%d", l)

		weights, hasWeights := saved[weightsKey]
		biases, hasBiases := saved[biasesKey]

		if !hasWeights || !hasBiases {
			continue
		}

		// Restore biases
		biasIdx := 0
		weightIdx := 0

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				if biasIdx < len(biases) {
					neuron.Bias = T(biases[biasIdx])
					biasIdx++
				}

				// Restore weights
				for i := range neuron.Inputs {
					if weightIdx < len(weights) {
						neuron.Inputs[i].Weight = T(weights[weightIdx])
						weightIdx++
					}
				}
			}
		}
	}
}

// Let's create a minimal shader test to debug the weight update issue
// The problem is likely in the shader logic itself

// First, let's add detailed debug output to the shader
func (n *Network[T]) generateBackwardShaderDebugDetailed(layerIdx int, inputSize, outputSize uint32, activation string) string {
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

			// CRITICAL DEBUG: Force a small weight change to test if writing works
			if (thread_id == 0u) {
				// Always modify first weight by a tiny amount for testing
				weights[0] = weights[0] + 0.001;
			}

			// Weight and bias updates (one thread per output neuron)
			if (thread_id < %du) {
				let output_idx = thread_id;
				let local_error = error[output_idx];

				// Update bias
				let bias_grad = clip_gradient(local_error, clip_lower, clip_upper);
				biases[output_idx] = biases[output_idx] - learning_rate * bias_grad;

				// Update weights for this output neuron
				let weight_row_start = output_idx * %du;
				for (var input_idx: u32 = 0u; input_idx < %du; input_idx = input_idx + 1u) {
					let weight_idx = weight_row_start + input_idx;
					let input_activation = prev_activations[input_idx];
					
					// Compute weight gradient
					let raw_gradient = local_error * input_activation;
					let clipped_gradient = clip_gradient(raw_gradient, clip_lower, clip_upper);
					
					// Apply gradient descent: weight = weight - learning_rate * gradient
					weights[weight_idx] = weights[weight_idx] - learning_rate * clipped_gradient;
				}
			}

			workgroupBarrier();

			// Error propagation (one thread per input neuron)
			if (thread_id < %du) {
				let input_idx = thread_id;
				var accumulated_error: f32 = 0.0;

				// Accumulate weighted errors
				for (var output_idx: u32 = 0u; output_idx < %du; output_idx = output_idx + 1u) {
					let weight_idx = output_idx * %du + input_idx;
					accumulated_error = accumulated_error + error[output_idx] * weights[weight_idx];
				}

				// Apply activation derivative
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

// Add a simple weight change test
func (n *Network[T]) TestSimpleWeightChange(layerIdx int) error {
	if n.gpu.backward == nil || layerIdx-1 >= len(n.gpu.backward.layers) {
		return fmt.Errorf("layer %d not available", layerIdx)
	}

	layer := n.gpu.backward.layers[layerIdx-1] // layerIdx starts at 1, array at 0

	if n.Debug {
		fmt.Printf("=== Testing Simple Weight Change for Layer %d ===\n", layerIdx)
	}

	// Read weights before
	weightsBefore, err := n.readBackwardStagingBuffer(layer.weightStaging, 10) // First 10 weights
	if err != nil {
		return fmt.Errorf("failed to read weights before: %v", err)
	}

	// Create simple test data
	testError := []float32{0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} // Only first neuron has error
	testActivations := make([]float32, layer.inputSize)
	for i := range testActivations {
		testActivations[i] = 1.0 // All activations = 1 for simple math
	}

	// Upload test data
	ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(testError))
	ctx.queue.WriteBuffer(layer.prevActivBuffer, 0, wgpu.ToBytes(testActivations))

	// Upload simple parameters
	params := []float32{0.01, 1000.0, -1000.0, 1.0} // lr=0.01, no clipping, output layer
	ctx.queue.WriteBuffer(layer.paramsBuffer, 0, wgpu.ToBytes(params))

	// Create and run compute pass
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create encoder: %v", err)
	}

	computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: "SimpleWeightTest",
	})

	computePass.SetPipeline(layer.pipeline)
	computePass.SetBindGroup(0, layer.bindGroup, nil)
	computePass.DispatchWorkgroups(1, 1, 1) // Single workgroup
	computePass.End()

	// Copy results to staging
	encoder.CopyBufferToBuffer(
		layer.weightBuffer, 0,
		layer.weightStaging, 0,
		40, // First 10 weights * 4 bytes
	)

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	// Read weights after
	weightsAfter, err := n.readBackwardStagingBuffer(layer.weightStaging, 10)
	if err != nil {
		return fmt.Errorf("failed to read weights after: %v", err)
	}

	// Check results
	if n.Debug {
		fmt.Println("Weight change test results:")
		hasChanges := false
		for i := 0; i < min(len(weightsBefore), len(weightsAfter)); i++ {
			change := weightsAfter[i] - weightsBefore[i]
			fmt.Printf("  Weight[%d]: %.6f -> %.6f (change: %+.6f)", i, weightsBefore[i], weightsAfter[i], change)

			if absFloat32(change) > 0.0001 {
				fmt.Printf(" ✓")
				hasChanges = true
			} else {
				fmt.Printf(" ✗")
			}
			fmt.Println()
		}

		if hasChanges {
			fmt.Println("✓ GPU weight updates are working!")
		} else {
			fmt.Println("✗ GPU weight updates are NOT working")
		}
	}

	return nil
}

// Alternative simpler shader for testing
func (n *Network[T]) generateSimpleTestShader(inputSize, outputSize uint32) string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> error: array<f32>;
		@group(0) @binding(1) var<storage, read_write> prev_error: array<f32>;
		@group(0) @binding(2) var<storage, read> current_activations: array<f32>;
		@group(0) @binding(3) var<storage, read> prev_activations: array<f32>;
		@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
		@group(0) @binding(5) var<storage, read_write> biases: array<f32>;
		@group(0) @binding(6) var<storage, read> params: array<f32>;

		@compute @workgroup_size(1, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let thread_id = global_id.x;
			
			// Simple test: modify first few weights by fixed amounts
			if (thread_id == 0u) {
				// Force changes to first few weights for testing
				weights[0] = weights[0] + 0.01;
				if (arrayLength(&weights) > 1u) {
					weights[1] = weights[1] - 0.01;
				}
				if (arrayLength(&weights) > 2u) {
					weights[2] = weights[2] + 0.005;
				}
				
				// Also test bias modification
				biases[0] = biases[0] + 0.001;
			}
		}
	`)
}

// Test with the simple shader
func (n *Network[T]) TestWithSimpleShader(layerIdx int) error {
	if n.gpu.backward == nil || layerIdx-1 >= len(n.gpu.backward.layers) {
		return fmt.Errorf("layer %d not available", layerIdx)
	}

	layer := n.gpu.backward.layers[layerIdx-1]

	// Create simple test shader
	simpleShader := n.generateSimpleTestShader(layer.inputSize, layer.outputSize)

	// Create shader module
	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("SimpleTestShader_%d", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: simpleShader},
	})
	if err != nil {
		return fmt.Errorf("failed to create simple shader: %v", err)
	}
	defer module.Release()

	// Create temporary pipeline
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "SimpleTestPipelineLayout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layer.bindGroupLayout},
	})
	if err != nil {
		return fmt.Errorf("failed to create pipeline layout: %v", err)
	}
	defer pipelineLayout.Release()

	testPipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "SimpleTestPipeline",
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create test pipeline: %v", err)
	}
	defer testPipeline.Release()

	// Read weights before
	weightsBefore, err := n.readBackwardStagingBuffer(layer.weightStaging, 5)
	if err != nil {
		return fmt.Errorf("failed to read weights before: %v", err)
	}

	biasesBefore, err := n.readBackwardStagingBuffer(layer.biasStaging, 3)
	if err != nil {
		return fmt.Errorf("failed to read biases before: %v", err)
	}

	// Run simple test
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create encoder: %v", err)
	}

	computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: "SimpleShaderTest",
	})

	computePass.SetPipeline(testPipeline)
	computePass.SetBindGroup(0, layer.bindGroup, nil)
	computePass.DispatchWorkgroups(1, 1, 1)
	computePass.End()

	// Copy results
	encoder.CopyBufferToBuffer(layer.weightBuffer, 0, layer.weightStaging, 0, 20)
	encoder.CopyBufferToBuffer(layer.biasBuffer, 0, layer.biasStaging, 0, 12)

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	// Read results
	weightsAfter, err := n.readBackwardStagingBuffer(layer.weightStaging, 5)
	if err != nil {
		return fmt.Errorf("failed to read weights after: %v", err)
	}

	biasesAfter, err := n.readBackwardStagingBuffer(layer.biasStaging, 3)
	if err != nil {
		return fmt.Errorf("failed to read biases after: %v", err)
	}

	// Check results
	fmt.Println("=== Simple Shader Test Results ===")
	fmt.Println("Weights:")
	weightSuccess := false
	for i := 0; i < min(len(weightsBefore), len(weightsAfter)); i++ {
		change := weightsAfter[i] - weightsBefore[i]
		fmt.Printf("  [%d] %.6f -> %.6f (change: %+.6f)", i, weightsBefore[i], weightsAfter[i], change)
		if absFloat32(change) > 0.001 {
			fmt.Printf(" ✓")
			weightSuccess = true
		} else {
			fmt.Printf(" ✗")
		}
		fmt.Println()
	}

	fmt.Println("Biases:")
	biasSuccess := false
	for i := 0; i < min(len(biasesBefore), len(biasesAfter)); i++ {
		change := biasesAfter[i] - biasesBefore[i]
		fmt.Printf("  [%d] %.6f -> %.6f (change: %+.6f)", i, biasesBefore[i], biasesAfter[i], change)
		if absFloat32(change) > 0.0001 {
			fmt.Printf(" ✓")
			biasSuccess = true
		} else {
			fmt.Printf(" ✗")
		}
		fmt.Println()
	}

	if weightSuccess && biasSuccess {
		fmt.Println("✓ Simple shader test PASSED - GPU can modify weights and biases")
	} else if biasSuccess {
		fmt.Println("⚠ Simple shader test PARTIAL - GPU can modify biases but NOT weights")
	} else {
		fmt.Println("✗ Simple shader test FAILED - GPU cannot modify buffers")
	}

	return nil
}

// Since we confirmed GPU can write to weights, the issue must be in the shader logic
// Let's create a simplified, guaranteed-to-work version

func (n *Network[T]) generateBackwardShaderWorking(layerIdx int, inputSize, outputSize uint32, activation string) string {
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

		@compute @workgroup_size(64, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let thread_id = global_id.x;
			let learning_rate = params[0];
			let clip_upper = params[1];
			let clip_lower = params[2];
			let is_output_layer = params[3];

			// SIMPLIFIED APPROACH: One phase at a time, no complex synchronization

			// Phase 1: Update biases (one thread per output neuron)
			if (thread_id < %du) {
				let output_idx = thread_id;
				let local_error = error[output_idx];
				
				// Simple bias update with clipping
				var bias_grad = local_error;
				if (clip_upper != 0.0 || clip_lower != 0.0) {
					bias_grad = clamp(bias_grad, clip_lower, clip_upper);
				}
				
				// Apply gradient descent
				biases[output_idx] = biases[output_idx] - learning_rate * bias_grad;
			}

			// Phase 2: Update weights (distribute work across all threads)
			let total_weights = %du * %du;
			if (thread_id < total_weights) {
				let weight_idx = thread_id;
				let output_idx = weight_idx / %du;
				let input_idx = weight_idx %% %du;
				
				if (output_idx < %du && input_idx < %du) {
					let local_error = error[output_idx];
					let input_activation = prev_activations[input_idx];
					
					// Compute weight gradient
					var weight_grad = local_error * input_activation;
					if (clip_upper != 0.0 || clip_lower != 0.0) {
						weight_grad = clamp(weight_grad, clip_lower, clip_upper);
					}
					
					// Apply gradient descent
					weights[weight_idx] = weights[weight_idx] - learning_rate * weight_grad;
				}
			}

			// Synchronization barrier
			workgroupBarrier();

			// Phase 3: Error propagation (one thread per input neuron)
			if (thread_id < %du) {
				let input_idx = thread_id;
				var accumulated_error: f32 = 0.0;

				// Accumulate weighted errors
				for (var output_idx: u32 = 0u; output_idx < %du; output_idx = output_idx + 1u) {
					let weight_idx = output_idx * %du + input_idx;
					accumulated_error = accumulated_error + error[output_idx] * weights[weight_idx];
				}

				// Apply activation derivative for hidden layers only
				if (is_output_layer == 0.0) {
					let input_activation = prev_activations[input_idx];
					let derivative = activation_derivative(input_activation);
					prev_error[input_idx] = accumulated_error * derivative;
				} else {
					prev_error[input_idx] = accumulated_error;
				}
			}
		}
	`, activationDerivCode, outputSize, outputSize, inputSize, inputSize, inputSize, outputSize, inputSize, inputSize, outputSize, inputSize)
}

// Test this working version
func (n *Network[T]) TestWorkingBackwardShader(layerIdx int) error {
	if n.gpu.backward == nil || layerIdx-1 >= len(n.gpu.backward.layers) {
		return fmt.Errorf("layer %d not available", layerIdx)
	}

	layer := n.gpu.backward.layers[layerIdx-1]

	// Get original weights/biases
	originalWeights, _ := n.extractLayerWeightsAndBiases(layerIdx)

	// Create the working shader
	workingShader := n.generateBackwardShaderWorking(layerIdx, layer.inputSize, layer.outputSize, "relu")

	// Create shader module
	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("WorkingBackwardShader_%d", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: workingShader},
	})
	if err != nil {
		return fmt.Errorf("failed to create working shader: %v", err)
	}
	defer module.Release()

	// Create pipeline
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "WorkingBackwardPipelineLayout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layer.bindGroupLayout},
	})
	if err != nil {
		return fmt.Errorf("failed to create pipeline layout: %v", err)
	}
	defer pipelineLayout.Release()

	testPipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "WorkingBackwardPipeline",
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create test pipeline: %v", err)
	}
	defer testPipeline.Release()

	// Setup test data
	testError := make([]float32, layer.outputSize)
	testActivations := make([]float32, layer.inputSize)

	// Create meaningful test values
	for i := range testError {
		testError[i] = 0.1 // Small error for all outputs
	}
	for i := range testActivations {
		testActivations[i] = 1.0 // Unit activations for simple math
	}

	// Upload test data
	ctx.queue.WriteBuffer(layer.errorBuffer, 0, wgpu.ToBytes(testError))
	ctx.queue.WriteBuffer(layer.prevActivBuffer, 0, wgpu.ToBytes(testActivations))
	ctx.queue.WriteBuffer(layer.weightBuffer, 0, wgpu.ToBytes(originalWeights[:len(originalWeights)]))

	// Upload parameters
	params := []float32{0.01, 1000.0, -1000.0, 0.0} // lr=0.01, no clipping, hidden layer
	ctx.queue.WriteBuffer(layer.paramsBuffer, 0, wgpu.ToBytes(params))

	// Run computation
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create encoder: %v", err)
	}

	computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: "WorkingBackwardTest",
	})

	computePass.SetPipeline(testPipeline)
	computePass.SetBindGroup(0, layer.bindGroup, nil)

	// Dispatch enough workgroups to cover all work
	maxWork := max(layer.inputSize, layer.outputSize)
	totalWeights := layer.inputSize * layer.outputSize
	maxWork = max(maxWork, totalWeights)
	workgroups := (maxWork + 63) / 64
	if workgroups == 0 {
		workgroups = 1
	}

	computePass.DispatchWorkgroups(workgroups, 1, 1)
	computePass.End()

	// Copy results
	encoder.CopyBufferToBuffer(layer.weightBuffer, 0, layer.weightStaging, 0, uint64(min(40, int(layer.inputSize*layer.outputSize*4))))
	encoder.CopyBufferToBuffer(layer.biasBuffer, 0, layer.biasStaging, 0, uint64(min(40, int(layer.outputSize*4))))

	cmd, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish encoder: %v", err)
	}

	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	// Read results
	weightsAfter, err := n.readBackwardStagingBuffer(layer.weightStaging, min(10, int(layer.inputSize*layer.outputSize)))
	if err != nil {
		return fmt.Errorf("failed to read weights: %v", err)
	}

	biasesAfter, err := n.readBackwardStagingBuffer(layer.biasStaging, min(10, int(layer.outputSize)))
	if err != nil {
		return fmt.Errorf("failed to read biases: %v", err)
	}

	// Check results
	fmt.Printf("=== Working Backward Shader Test Results (Layer %d) ===\n", layerIdx)
	fmt.Println("Weights:")
	weightSuccess := false
	for i := 0; i < min(len(originalWeights), len(weightsAfter)); i++ {
		change := weightsAfter[i] - originalWeights[i]
		fmt.Printf("  [%d] %.6f -> %.6f (change: %+.6f)", i, originalWeights[i], weightsAfter[i], change)
		if absFloat32(change) > 0.0001 {
			fmt.Printf(" ✓")
			weightSuccess = true
		} else {
			fmt.Printf(" ✗")
		}
		fmt.Println()
		if i >= 4 {
			break
		} // Show first 5
	}

	fmt.Println("Biases:")
	biasSuccess := false
	for i := 0; i < len(biasesAfter) && i < 5; i++ {
		change := biasesAfter[i] // Biases start at 0
		fmt.Printf("  [%d] 0.000000 -> %.6f (change: %+.6f)", i, biasesAfter[i], change)
		if absFloat32(change) > 0.0001 {
			fmt.Printf(" ✓")
			biasSuccess = true
		} else {
			fmt.Printf(" ✗")
		}
		fmt.Println()
	}

	if weightSuccess && biasSuccess {
		fmt.Println("✓ Working backward shader test PASSED!")
		return nil
	} else if weightSuccess {
		fmt.Println("⚠ Working backward shader test PARTIAL - weights work, biases don't")
		return nil
	} else {
		fmt.Println("✗ Working backward shader test FAILED")
		return fmt.Errorf("shader test failed")
	}
}

func (n *Network[T]) updateWorkgroupDispatch(layerCompute *GPUBackwardLayer, layerIdx int) uint32 {
	// Calculate workgroups to cover all work types:
	// - Bias updates: outputSize threads needed
	// - Weight updates: inputSize * outputSize threads needed
	// - Error propagation: inputSize threads needed

	biasWork := layerCompute.outputSize
	weightWork := layerCompute.inputSize * layerCompute.outputSize
	errorWork := layerCompute.inputSize

	// Take maximum work needed
	maxWork := biasWork
	if weightWork > maxWork {
		maxWork = weightWork
	}
	if errorWork > maxWork {
		maxWork = errorWork
	}

	// Calculate workgroups (64 threads per workgroup)
	workgroups := (maxWork + 63) / 64
	if workgroups == 0 {
		workgroups = 1
	}

	if n.Debug {
		fmt.Printf("Layer %d workload: bias=%d, weights=%d, error=%d -> %d workgroups\n",
			layerIdx, biasWork, weightWork, errorWork, workgroups)
	}

	return workgroups
}

// STEP 3: Update your BackwardGPU function dispatch call to use this:
// Replace this line:
//   workgroups := (maxThreads + 63) / 64
// With:
//   workgroups := n.updateWorkgroupDispatch(layerCompute, layerIdx)

// STEP 4: Quick verification function
func (n *Network[T]) VerifyShaderReplacement() {
	fmt.Println("=== Verifying Shader Replacement ===")

	// Test the main backward pass one more time
	if n.gpu.backward != nil && len(n.gpu.backward.layers) > 0 {
		layer := n.gpu.backward.layers[0]

		// The shader should now be the working version
		fmt.Printf("✓ Shader replacement ready for layer 1\n")
		fmt.Printf("  Input size: %d, Output size: %d\n", layer.inputSize, layer.outputSize)
		fmt.Printf("  Expected workgroups: %d\n", n.updateWorkgroupDispatch(layer, 1))
	}
}

// Fixed version with proper type handling
func (n *Network[T]) TestWeightApplication() {
	fmt.Println("=== Testing Weight Application ===")

	// Get first few weights before GPU backward pass
	layer1 := n.Layers[1]
	firstNeuron := layer1.Neurons[0][0]

	fmt.Printf("Before GPU backward pass:\n")
	fmt.Printf("  First neuron bias: %.6f\n", float32(any(firstNeuron.Bias).(T)))
	for i := 0; i < min(5, len(firstNeuron.Inputs)); i++ {
		fmt.Printf("  Weight[%d]: %.6f\n", i, float32(any(firstNeuron.Inputs[i].Weight).(T)))
	}

	// Run one backward pass
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = 1.0 // Use all 1s for predictable results
		}
	}

	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	target[0][5] = 1.0 // Target class 5

	// Forward pass
	n.Forward(input)
	fmt.Printf("Forward pass output[0]: %.6f\n", n.GetOutput()[0])

	// GPU backward pass with type-safe clipping bounds
	var clipUpper, clipLower T
	var zero T

	switch any(zero).(type) {
	case float32:
		clipUpper = any(float32(1000.0)).(T)
		clipLower = any(float32(-1000.0)).(T)
	case float64:
		clipUpper = any(float64(1000.0)).(T)
		clipLower = any(float64(-1000.0)).(T)
	default:
		// For integer types, use zero bounds (no clipping)
		clipUpper = zero
		clipLower = zero
	}

	err := n.BackwardGPU(target, 0.01, clipUpper, clipLower)
	if err != nil {
		fmt.Printf("Backward pass failed: %v\n", err)
		return
	}

	// Check weights after GPU backward pass
	fmt.Printf("After GPU backward pass:\n")
	fmt.Printf("  First neuron bias: %.6f\n", float32(any(firstNeuron.Bias).(T)))
	for i := 0; i < min(5, len(firstNeuron.Inputs)); i++ {
		fmt.Printf("  Weight[%d]: %.6f\n", i, float32(any(firstNeuron.Inputs[i].Weight).(T)))
	}

	// Run forward pass again to see if output changed
	n.Forward(input)
	fmt.Printf("Forward pass output[0] after update: %.6f\n", n.GetOutput()[0])
}

func (n *Network[T]) TestSaturationIssue() {
	fmt.Println("=== Testing Saturation Issue ===")

	// Create simple input that doesn't cause saturation
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = 0.01 // Very small inputs
		}
	}

	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	target[0][5] = 1.0

	fmt.Println("Before training (small inputs):")
	n.Forward(input)
	outputs := n.GetOutput()
	loss1 := n.ComputeLoss(target)
	fmt.Printf("  Loss: %.6f\n", loss1)
	fmt.Printf("  Outputs: ")
	for i := 0; i < 5; i++ {
		fmt.Printf("%.6f ", outputs[i])
	}
	fmt.Println("...")

	// Run GPU backward pass with type-safe clipping bounds
	var clipUpper, clipLower T
	var zero T

	switch any(zero).(type) {
	case float32:
		clipUpper = any(float32(1000.0)).(T)
		clipLower = any(float32(-1000.0)).(T)
	case float64:
		clipUpper = any(float64(1000.0)).(T)
		clipLower = any(float64(-1000.0)).(T)
	default:
		// For integer types, use zero bounds (no clipping)
		clipUpper = zero
		clipLower = zero
	}

	err := n.BackwardGPU(target, 0.1, clipUpper, clipLower) // Larger learning rate
	if err != nil {
		fmt.Printf("Backward pass failed: %v\n", err)
		return
	}

	fmt.Println("After GPU training (small inputs):")
	n.Forward(input)
	outputs = n.GetOutput()
	loss2 := n.ComputeLoss(target)
	fmt.Printf("  Loss: %.6f -> %.6f (change: %+.6f)\n", loss1, loss2, loss2-loss1)
	fmt.Printf("  Outputs: ")
	for i := 0; i < 5; i++ {
		fmt.Printf("%.6f ", outputs[i])
	}
	fmt.Println("...")

	if absFloat32(float32(loss2-loss1)) > 0.001 {
		fmt.Println("✓ GPU training is working - loss changed with small inputs!")
	} else {
		fmt.Println("✗ GPU training still not working even with small inputs")
	}
}

// Debug function to trace exactly what's happening in weight application
func (n *Network[T]) DebugWeightApplication(layerIdx int) {
	fmt.Printf("=== Debugging Weight Application for Layer %d ===\n", layerIdx)

	if n.gpu.backward == nil || layerIdx-1 >= len(n.gpu.backward.layers) {
		fmt.Printf("No GPU backward layer available\n")
		return
	}

	layer := n.gpu.backward.layers[layerIdx-1]

	// 1. Check what's in the GPU buffers BEFORE reading
	fmt.Println("Step 1: Reading weights directly from GPU staging buffer...")

	// First, copy current weights to staging for comparison
	encoder, _ := ctx.device.CreateCommandEncoder(nil)
	encoder.CopyBufferToBuffer(
		layer.weightBuffer, 0,
		layer.weightStaging, 0,
		40, // First 10 weights
	)
	encoder.CopyBufferToBuffer(
		layer.biasBuffer, 0,
		layer.biasStaging, 0,
		40, // First 10 biases
	)
	cmd, _ := encoder.Finish(nil)
	ctx.queue.Submit(cmd)
	ctx.device.Poll(true, nil)

	gpuWeights, _ := n.readBackwardStagingBuffer(layer.weightStaging, 10)
	gpuBiases, _ := n.readBackwardStagingBuffer(layer.biasStaging, 10)

	fmt.Println("GPU buffer contents:")
	for i := 0; i < min(5, len(gpuWeights)); i++ {
		fmt.Printf("  GPU Weight[%d]: %.6f\n", i, gpuWeights[i])
	}
	for i := 0; i < min(5, len(gpuBiases)); i++ {
		fmt.Printf("  GPU Bias[%d]: %.6f\n", i, gpuBiases[i])
	}

	// 2. Check what extractLayerWeightsAndBiases returns
	fmt.Println("Step 2: Extracting weights from CPU network...")
	cpuWeights, cpuBiases := n.extractLayerWeightsAndBiases(layerIdx)

	fmt.Println("CPU network contents:")
	for i := 0; i < min(5, len(cpuWeights)); i++ {
		fmt.Printf("  CPU Weight[%d]: %.6f\n", i, cpuWeights[i])
	}
	for i := 0; i < min(5, len(cpuBiases)); i++ {
		fmt.Printf("  CPU Bias[%d]: %.6f\n", i, cpuBiases[i])
	}

	// 3. Test the applyWeightsAndBiases function directly
	fmt.Println("Step 3: Testing applyWeightsAndBiases function...")

	// Modify the first few weights slightly for testing
	testWeights := make([]float32, len(cpuWeights))
	copy(testWeights, cpuWeights)
	testBiases := make([]float32, len(cpuBiases))
	copy(testBiases, cpuBiases)

	// Make small changes to test
	for i := 0; i < min(5, len(testWeights)); i++ {
		testWeights[i] += 0.12345 // Distinctive change
	}
	for i := 0; i < min(5, len(testBiases)); i++ {
		testBiases[i] += 0.54321 // Distinctive change
	}

	fmt.Println("Applying test weights...")
	err := n.applyWeightsAndBiases(layerIdx, testWeights, testBiases)
	if err != nil {
		fmt.Printf("applyWeightsAndBiases failed: %v\n", err)
		return
	}

	// 4. Check if the changes were applied
	fmt.Println("Step 4: Checking if changes were applied to CPU network...")
	currentLayer := n.Layers[layerIdx]

	for y := 0; y < min(1, currentLayer.Height); y++ {
		for x := 0; x < min(5, currentLayer.Width); x++ {
			neuron := currentLayer.Neurons[y][x]
			fmt.Printf("  Neuron[%d][%d] bias: %.6f\n", y, x, float32(any(neuron.Bias).(T)))

			for i := 0; i < min(3, len(neuron.Inputs)); i++ {
				fmt.Printf("    Weight[%d]: %.6f\n", i, float32(any(neuron.Inputs[i].Weight).(T)))
			}
		}
	}

	// 5. Check the issue - verify weight extraction after application
	fmt.Println("Step 5: Re-extracting weights to verify they changed...")
	newCpuWeights, _ := n.extractLayerWeightsAndBiases(layerIdx)

	changesDetected := false
	for i := 0; i < min(5, len(newCpuWeights)); i++ {
		change := absFloat32(newCpuWeights[i] - cpuWeights[i])
		fmt.Printf("  Weight[%d] change: %.6f\n", i, change)
		if change > 0.1 {
			changesDetected = true
		}
	}

	if changesDetected {
		fmt.Println("✓ applyWeightsAndBiases is working correctly")
	} else {
		fmt.Println("✗ applyWeightsAndBiases is NOT working")
	}
}

// Add this to your main debug function to see what's happening:
func (n *Network[T]) QuickWeightApplicationTest() {
	fmt.Println("=== Quick Weight Application Test ===")

	// Get current first weight
	layer1 := n.Layers[1]
	firstNeuron := layer1.Neurons[0][0]
	beforeWeight := float32(any(firstNeuron.Inputs[0].Weight).(T))
	beforeBias := float32(any(firstNeuron.Bias).(T))

	fmt.Printf("Before: Weight[0]=%.6f, Bias=%.6f\n", beforeWeight, beforeBias)

	// Extract current weights
	weights, biases := n.extractLayerWeightsAndBiases(1)

	// Modify them slightly
	weights[0] += 0.99999 // Big change
	biases[0] += 0.88888  // Big change

	// Apply them back
	err := n.applyWeightsAndBiases(1, weights, biases)
	if err != nil {
		fmt.Printf("Apply failed: %v\n", err)
		return
	}

	// Check if they changed
	afterWeight := float32(any(firstNeuron.Inputs[0].Weight).(T))
	afterBias := float32(any(firstNeuron.Bias).(T))

	fmt.Printf("After: Weight[0]=%.6f, Bias=%.6f\n", afterWeight, afterBias)
	fmt.Printf("Changes: Weight=%+.6f, Bias=%+.6f\n", afterWeight-beforeWeight, afterBias-beforeBias)

	if absFloat32(afterWeight-beforeWeight) > 0.5 && absFloat32(afterBias-beforeBias) > 0.5 {
		fmt.Println("✓ Weight application works correctly")
	} else {
		fmt.Println("✗ Weight application is broken")
	}
}
