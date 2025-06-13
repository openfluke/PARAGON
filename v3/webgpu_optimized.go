// webgpu_optimized.go
package paragon

import (
	"fmt"
	"reflect"

	"github.com/openfluke/webgpu/wgpu"
)

// GPULayerCompute represents a single layer's GPU computation
type GPULayerCompute struct {
	pipeline            *wgpu.ComputePipeline
	bindGroup           *wgpu.BindGroup
	bindGroupLayout     *wgpu.BindGroupLayout
	inputBuffer         *wgpu.Buffer
	outputBuffer        *wgpu.Buffer
	weightBuffer        *wgpu.Buffer
	biasBuffer          *wgpu.Buffer
	stagingBuffer       *wgpu.Buffer
	derivBuffer         *wgpu.Buffer
	gradBuffer          *wgpu.Buffer
	weightStagingBuffer *wgpu.Buffer
	biasStagingBuffer   *wgpu.Buffer
	// Store uniform buffers as part of the layer compute
	lrBuffer        *wgpu.Buffer
	clipUpperBuffer *wgpu.Buffer
	clipLowerBuffer *wgpu.Buffer
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
	typ := reflect.TypeOf(*new(T)).Kind()

	switch typ {
	case reflect.Float32, reflect.Int32, reflect.Uint32:
		// Supported GPU types
	default:
		return fmt.Errorf("❌ WebGPU acceleration only supports f32, i32, or u32 numeric types — got: %v", typ)
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
			// Clean up already created layers
			for _, lc := range n.gpu.optimized.layers {
				lc.cleanup()
			}
			n.gpu.optimized = nil
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

	shaderCode := n.generateLayerShader(layerIdx, inputSize, outputSize)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create shader module: %v", err)
	}
	defer module.Release()

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
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    4,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    5,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    6,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeUniform,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    7,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeUniform,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    8,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeUniform,
					HasDynamicOffset: false,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bind group layout: %v", err)
	}

	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, fmt.Errorf("failed to create pipeline layout: %v", err)
	}
	defer pipelineLayout.Release()

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

	layerCompute := &GPULayerCompute{
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
		inputSize:       inputSize,
		outputSize:      outputSize,
		layerIndex:      layerIdx,
	}

	// Initialize buffers
	if err := layerCompute.initializeBuffers(ctx.device, layerIdx, inputSize, outputSize); err != nil {
		layerCompute.cleanup()
		return nil, err
	}

	// Initialize weights and biases
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)
	if len(weights) != int(inputSize*outputSize) || len(biases) != int(outputSize) {
		layerCompute.cleanup()
		return nil, fmt.Errorf("invalid weight or bias size for layer %d: weights=%d, biases=%d", layerIdx, len(weights), len(biases))
	}
	ctx.queue.WriteBuffer(layerCompute.weightBuffer, 0, wgpu.ToBytes(weights))
	ctx.queue.WriteBuffer(layerCompute.biasBuffer, 0, wgpu.ToBytes(biases))

	// Create uniform buffers (these need to be kept alive)
	layerCompute.lrBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_LR", layerIdx),
		Size:  4,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create learning rate buffer: %v", err)
	}

	layerCompute.clipUpperBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_ClipUpper", layerIdx),
		Size:  4,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create clip upper buffer: %v", err)
	}

	layerCompute.clipLowerBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_ClipLower", layerIdx),
		Size:  4,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create clip lower buffer: %v", err)
	}

	// Initialize uniform buffers with default values
	ctx.queue.WriteBuffer(layerCompute.lrBuffer, 0, wgpu.ToBytes([]float32{0.01}))
	ctx.queue.WriteBuffer(layerCompute.clipUpperBuffer, 0, wgpu.ToBytes([]float32{1.0}))
	ctx.queue.WriteBuffer(layerCompute.clipLowerBuffer, 0, wgpu.ToBytes([]float32{-1.0}))

	// Create bind group with all buffers
	layerCompute.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_BindGroup", layerIdx),
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: layerCompute.inputBuffer, Size: layerCompute.inputBuffer.GetSize()},
			{Binding: 1, Buffer: layerCompute.outputBuffer, Size: layerCompute.outputBuffer.GetSize()},
			{Binding: 2, Buffer: layerCompute.weightBuffer, Size: layerCompute.weightBuffer.GetSize()},
			{Binding: 3, Buffer: layerCompute.biasBuffer, Size: layerCompute.biasBuffer.GetSize()},
			{Binding: 4, Buffer: layerCompute.derivBuffer, Size: layerCompute.derivBuffer.GetSize()},
			{Binding: 5, Buffer: layerCompute.gradBuffer, Size: layerCompute.gradBuffer.GetSize()},
			{Binding: 6, Buffer: layerCompute.lrBuffer, Size: 4},
			{Binding: 7, Buffer: layerCompute.clipUpperBuffer, Size: 4},
			{Binding: 8, Buffer: layerCompute.clipLowerBuffer, Size: 4},
		},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bind group: %v", err)
	}

	layerCompute.workgroupsX = (outputSize + 255) / 256
	layerCompute.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created layer %d compute: %dx%d -> %dx%d, workgroups: %d\n",
			layerIdx, prevLayer.Width, prevLayer.Height,
			currentLayer.Width, currentLayer.Height, layerCompute.workgroupsX)
	}

	return layerCompute, nil
}

// Initialize all buffers for a layer
func (lc *GPULayerCompute) initializeBuffers(device *wgpu.Device, layerIdx int, inputSize, outputSize uint32) error {
	var err error

	lc.inputBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Input", layerIdx),
		Size:  uint64(inputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create input buffer: %v", err)
	}

	lc.outputBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Output", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create output buffer: %v", err)
	}

	lc.weightBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Weights", layerIdx),
		Size:  uint64(inputSize*outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create weight buffer: %v", err)
	}

	lc.biasBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Biases", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create bias buffer: %v", err)
	}

	lc.derivBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Deriv", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create derivative buffer: %v", err)
	}

	lc.gradBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Grad", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create gradient buffer: %v", err)
	}

	lc.weightStagingBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Weight_Staging", layerIdx),
		Size:  uint64(inputSize*outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create weight staging buffer: %v", err)
	}

	lc.biasStagingBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Bias_Staging", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create bias staging buffer: %v", err)
	}

	lc.stagingBuffer, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Staging", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create staging buffer: %v", err)
	}

	return nil
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
	if lc.derivBuffer != nil {
		lc.derivBuffer.Destroy()
	}
	if lc.gradBuffer != nil {
		lc.gradBuffer.Destroy()
	}
	if lc.weightStagingBuffer != nil {
		lc.weightStagingBuffer.Destroy()
	}
	if lc.biasStagingBuffer != nil {
		lc.biasStagingBuffer.Destroy()
	}
	if lc.stagingBuffer != nil {
		lc.stagingBuffer.Destroy()
	}
	if lc.lrBuffer != nil {
		lc.lrBuffer.Destroy()
	}
	if lc.clipUpperBuffer != nil {
		lc.clipUpperBuffer.Destroy()
	}
	if lc.clipLowerBuffer != nil {
		lc.clipLowerBuffer.Destroy()
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
	typ := n.gpu.wgslType
	activationCode, derivativeCode := getActivationAndDerivativeCode(activation, typ)

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input: array<%s>;
		@group(0) @binding(1) var<storage, read_write> output: array<%s>;
		@group(0) @binding(2) var<storage, read_write> weights: array<%s>;
		@group(0) @binding(3) var<storage, read_write> biases: array<%s>;
		@group(0) @binding(4) var<storage, read_write> derivatives: array<%s>;
		@group(0) @binding(5) var<storage, read_write> gradients: array<%s>;
		@group(0) @binding(6) var<uniform> lr: f32;
		@group(0) @binding(7) var<uniform> clip_upper: f32;
		@group(0) @binding(8) var<uniform> clip_lower: f32;

		%s
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

			let activated = activate(sum);
			output[output_idx] = activated;
			derivatives[output_idx] = derivative(activated);
		}

		@compute @workgroup_size(256, 1, 1)
		fn backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let output_idx = global_id.x;

			if (output_idx >= %du) {
				return;
			}

			let grad = gradients[output_idx];
			let deriv = derivatives[output_idx];
			let error = grad * deriv;
			let weight_offset = output_idx * %du;

			let grad_b = clamp(error, clip_lower, clip_upper);
			biases[output_idx] += lr * grad_b;

			for (var i: u32 = 0u; i < %du; i++) {
				let grad_w = clamp(error * input[i], clip_lower, clip_upper);
				weights[weight_offset + i] += lr * grad_w;
			}

			if (%du > 0u) {
				for (var i: u32 = 0u; i < %du; i++) {
					let prev_grad_idx = i;
					if (prev_grad_idx < %du) {
						gradients[prev_grad_idx] += error * weights[weight_offset + i];
					}
				}
			}
		}
	`, typ, typ, typ, typ, typ, typ, activationCode, derivativeCode,
		outputSize, typ, inputSize, inputSize,
		outputSize, inputSize, inputSize, inputSize, inputSize, inputSize)
}

func getActivationAndDerivativeCode(activation, typ string) (string, string) {
	switch activation {
	case "softmax":
		if typ == "f32" {
			return `
				fn activate(x: f32) -> f32 { return x; }
			`, `
				fn derivative(x: f32) -> f32 { return 1.0; }
			`
		}
		if typ == "u32" {
			return `
				fn activate(x: u32) -> u32 { return x; }
			`, `
				fn derivative(x: u32) -> u32 { return 1u; }
			`
		}
		return `
			fn activate(x: i32) -> i32 { return x; }
		`, `
			fn derivative(x: i32) -> i32 { return i32(1); }
		`

	case "relu":
		return fmt.Sprintf(`
			fn activate(x: %s) -> %s { return max(%s(0), x); }
		`, typ, typ, typ),
			fmt.Sprintf(`
			fn derivative(x: %s) -> %s {
				return select(%s(0), %s(1), x > %s(0));
			}
		`, typ, typ, typ, typ, typ)

	case "leaky_relu":
		if typ == "f32" {
			return `
				fn activate(x: f32) -> f32 { return select(0.01 * x, x, x > 0.0); }
			`, `
				fn derivative(x: f32) -> f32 { return select(0.01, 1.0, x > 0.0); }
			`
		}
		if typ == "u32" {
			return `
				fn activate(x: u32) -> u32 { return x; }
			`, `
				fn derivative(x: u32) -> u32 { return 1u; }
			`
		}
		return `
			fn activate(x: i32) -> i32 {
				if (x >= i32(0)) { return x; }
				var leak = x / i32(100);
				if (leak == i32(0) && x < i32(0)) { leak = i32(-1); }
				return leak;
			}
		`, `
			fn derivative(x: i32) -> i32 {
				return select(i32(1), i32(100), x <= i32(0));
			}
		`

	case "elu":
		if typ == "f32" {
			return `
				fn activate(x: f32) -> f32 {
					if (x >= 0.0) { return x; }
					return exp(max(x, -10.0)) - 1.0;
				}
			`, `
				fn derivative(x: f32) -> f32 {
					if (x >= 0.0) { return 1.0; }
					return exp(max(x, -10.0));
				}
			`
		}
		if typ == "u32" {
			return `
				fn activate(x: u32) -> u32 { return x; }
			`, `
				fn derivative(x: u32) -> u32 { return 1u; }
			`
		}
		return `
			fn activate(x: i32) -> i32 {
				if (x >= i32(0)) { return x; }
				let scale = i32(2147483647);
				if (x <= -scale) { return -scale; }
				return x / i32(2);
			}
		`, `
			fn derivative(x: i32) -> i32 { return 1; }
		`

	case "tanh":
		const tanhApprox = `
			fn tanh_approx(input: f32) -> f32 {
				if (input > 1.0) { return 1.0; }
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
				fn activate(x: f32) -> f32 { return tanh_approx(x); }
			`, tanhApprox + `
				fn derivative(x: f32) -> f32 {
					let t = tanh_approx(x);
					return 1.0 - t * t;
				}
			`
		}
		if typ == "i32" {
			return tanhApprox + `
				fn activate(x: i32) -> i32 { return i32(tanh_approx(f32(x))); }
			`, tanhApprox + `
				fn derivative(x: i32) -> i32 {
					let t = tanh_approx(f32(x));
					return i32(1.0 - t * t);
				}
			`
		}
		return tanhApprox + `
			fn activate(x: u32) -> u32 {
				let t = tanh_approx(f32(x));
				return u32(max(t, 0.0));
			}
		`, `
			fn derivative(x: u32) -> u32 { return 1u; }
		`

	case "sigmoid":
		if typ == "f32" {
			return `
				fn activate(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
			`, `
				fn derivative(x: f32) -> f32 {
					let s = 1.0 / (1.0 + exp(-x));
					return s * (1.0 - s);
				}
			`
		}
		if typ == "u32" {
			return `
				fn activate(x: u32) -> u32 {
					let scaled = f32(x) / f32(2147483647);
					let s = 1.0 / (1.0 + exp(-scaled * 0.5));
					return u32(round(s * f32(2147483647) * 2.0));
				}
			`, `
				fn derivative(x: u32) -> u32 {
					let scaled = f32(x) / f32(2147483647);
					let s = 1.0 / (1.0 + exp(-scaled * 0.5));
					return u32(round((s * (1.0 - s)) * f32(2147483647)));
				}
			`
		}
		return `
			fn activate(x: i32) -> i32 {
				let scaled = f32(x) / f32(2147483647);
				let s = 1.0 / (1.0 + exp(-scaled));
				return i32(round(s * f32(2147483647)));
			}
		`, `
			fn derivative(x: i32) -> i32 {
				let scaled = f32(x) / f32(2147483647);
				let s = 1.0 / (1.0 + exp(-scaled));
				return i32(round((s * (1.0 - s)) * f32(2147483647)));
			}
		`

	default:
		return fmt.Sprintf(`
			fn activate(x: %s) -> %s { return x; }
		`, typ, typ),
			fmt.Sprintf(`
			fn derivative(x: %s) -> %s { return %s(1); }
		`, typ, typ, typ)
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

	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			biases[neuronIdx] = T(any(neuron.Bias).(T))

			weightOffset := neuronIdx * inputSize
			for i, conn := range neuron.Inputs {
				if i < inputSize {
					weights[weightOffset+i] = T(any(conn.Weight).(T))
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

	if len(inputs) == 0 || len(inputs[0]) == 0 {
		return fmt.Errorf("invalid input dimensions")
	}

	// Convert inputs to correct type
	var inputData []T
	inputData = make([]T, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, T(val))
		}
	}

	// Write initial input data
	if len(n.gpu.optimized.layers) > 0 {
		firstLayer := n.gpu.optimized.layers[0]
		ctx.queue.WriteBuffer(firstLayer.inputBuffer, 0, wgpu.ToBytes(inputData))
	}

	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Process each layer
	for i, layerCompute := range n.gpu.optimized.layers {
		// Create compute pass
		computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Compute", i+1),
		})

		computePass.SetPipeline(layerCompute.pipeline)
		computePass.SetBindGroup(0, layerCompute.bindGroup, nil)
		computePass.DispatchWorkgroups(layerCompute.workgroupsX, layerCompute.workgroupsY, 1)
		computePass.End()

		// Copy output to staging buffer for the last layer
		if i == len(n.gpu.optimized.layers)-1 {
			encoder.CopyBufferToBuffer(
				layerCompute.outputBuffer, 0,
				layerCompute.stagingBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}

		// Copy output to next layer's input (if not the last layer)
		if i < len(n.gpu.optimized.layers)-1 {
			nextLayerCompute := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				layerCompute.outputBuffer, 0,
				nextLayerCompute.inputBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}
	}

	// Finish and submit command buffer
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		encoder.Release()
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)
	commandBuffer.Release()
	encoder.Release()

	// Wait for GPU operations to complete
	ctx.device.Poll(true, nil)

	// Read final output
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readStagingBuffer(finalLayer.stagingBuffer, int(finalLayer.outputSize))
	if err != nil {
		return fmt.Errorf("failed to read final output: %v", err)
	}

	// Apply output to network
	n.applyFinalOutput(finalOutput)
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

	// Wait for mapping with timeout
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
		}
	}

readData:
	data := buffer.GetMappedRange(0, uint(size*4))
	if data == nil {
		buffer.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	result := make([]T, size)
	copy(result, wgpu.FromBytes[T](data))
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
	if outputLayer.Neurons[0][0].Activation == "softmax" {
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

func (n *Network[T]) Backward(targets [][]float64, lr float64, clipUpper T, clipLower T) {
	// Use optimized GPU if available and enabled
	if n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized {
		err := n.backwardGPU(targets, lr, clipUpper, clipLower)
		if err != nil {
			if n.Debug {
				fmt.Printf("Optimized GPU backward failed, falling back to CPU: %v\n", err)
			}
			// Fall back to CPU
			n.backwardCPU(targets, lr, clipUpper, clipLower)
		}
		return
	}

	// Fallback to existing implementation
	n.backwardCPU(targets, lr, clipUpper, clipLower)
}

// GPU backward pass implementation
func (n *Network[T]) backwardGPU(targets [][]float64, lr float64, clipUpper T, clipLower T) error {
	if !n.gpu.optimized.initialized {
		return fmt.Errorf("optimized GPU not initialized")
	}

	// Compute output layer errors
	outputLayer := n.Layers[n.OutputLayer]
	outputSize := uint32(outputLayer.Width * outputLayer.Height)
	errors := make([]float32, outputSize)
	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			pred := float64(outputLayer.Neurons[y][x].Value)
			targ := targets[y][x]
			errors[idx] = float32(targ - pred)
			idx++
		}
	}

	// Update uniform values for all layers
	for _, layerCompute := range n.gpu.optimized.layers {
		ctx.queue.WriteBuffer(layerCompute.lrBuffer, 0, wgpu.ToBytes([]float32{float32(lr)}))
		ctx.queue.WriteBuffer(layerCompute.clipUpperBuffer, 0, wgpu.ToBytes([]float32{float32(clipUpper)}))
		ctx.queue.WriteBuffer(layerCompute.clipLowerBuffer, 0, wgpu.ToBytes([]float32{float32(clipLower)}))
	}

	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Process layers in reverse order
	for i := len(n.gpu.optimized.layers) - 1; i >= 0; i-- {
		layerCompute := n.gpu.optimized.layers[i]

		// Write gradient buffer
		ctx.queue.WriteBuffer(layerCompute.gradBuffer, 0, wgpu.ToBytes(errors))

		// Create backward compute pass
		module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
			Label:          fmt.Sprintf("Layer_%d_Backward_Shader", layerCompute.layerIndex),
			WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: n.generateLayerShader(layerCompute.layerIndex, layerCompute.inputSize, layerCompute.outputSize)},
		})
		if err != nil {
			encoder.Release()
			return fmt.Errorf("failed to create backward shader module: %v", err)
		}
		defer module.Release()

		pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
			Label:            fmt.Sprintf("Layer_%d_Backward_PipelineLayout", layerCompute.layerIndex),
			BindGroupLayouts: []*wgpu.BindGroupLayout{layerCompute.bindGroupLayout},
		})
		if err != nil {
			encoder.Release()
			return fmt.Errorf("failed to create backward pipeline layout: %v", err)
		}
		defer pipelineLayout.Release()

		pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
			Label:  fmt.Sprintf("Layer_%d_Backward_Pipeline", layerCompute.layerIndex),
			Layout: pipelineLayout,
			Compute: wgpu.ProgrammableStageDescriptor{
				Module:     module,
				EntryPoint: "backward_main",
			},
		})
		if err != nil {
			encoder.Release()
			return fmt.Errorf("failed to create backward pipeline: %v", err)
		}
		defer pipeline.Release()

		computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Backward_Compute", layerCompute.layerIndex),
		})
		computePass.SetPipeline(pipeline)
		computePass.SetBindGroup(0, layerCompute.bindGroup, nil)
		computePass.DispatchWorkgroups(layerCompute.workgroupsX, layerCompute.workgroupsY, 1)
		computePass.End()

		// Copy updated weights and biases to staging buffers
		encoder.CopyBufferToBuffer(
			layerCompute.weightBuffer, 0,
			layerCompute.weightStagingBuffer, 0,
			uint64(layerCompute.inputSize*layerCompute.outputSize)*4,
		)
		encoder.CopyBufferToBuffer(
			layerCompute.biasBuffer, 0,
			layerCompute.biasStagingBuffer, 0,
			uint64(layerCompute.outputSize)*4,
		)

		// Copy gradients for next layer (if not first layer)
		if i > 0 {
			encoder.CopyBufferToBuffer(
				layerCompute.gradBuffer, 0,
				layerCompute.stagingBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}
	}

	// Finish and submit
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		encoder.Release()
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)
	commandBuffer.Release()
	encoder.Release()

	// Wait for GPU operations
	ctx.device.Poll(true, nil)

	// Read back updated weights and biases
	for i, layerCompute := range n.gpu.optimized.layers {
		weights, err := n.readStagingBuffer(layerCompute.weightStagingBuffer, int(layerCompute.inputSize*layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read weights for layer %d: %v", i+1, err)
		}

		biases, err := n.readStagingBuffer(layerCompute.biasStagingBuffer, int(layerCompute.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read biases for layer %d: %v", i+1, err)
		}

		// Update network neurons
		currentLayer := n.Layers[layerCompute.layerIndex]
		prevLayer := n.Layers[layerCompute.layerIndex-1]
		inputSize := prevLayer.Width * prevLayer.Height

		for y := 0; y < currentLayer.Height; y++ {
			for x := 0; x < currentLayer.Width; x++ {
				neuronIdx := y*currentLayer.Width + x
				neuron := currentLayer.Neurons[y][x]
				weightOffset := neuronIdx * inputSize

				// Update weights
				for j := range neuron.Inputs {
					if j < inputSize && weightOffset+j < len(weights) {
						neuron.Inputs[j].Weight = weights[weightOffset+j]
					}
				}

				// Update bias
				if neuronIdx < len(biases) {
					neuron.Bias = biases[neuronIdx]
				}
			}
		}

		// Prepare errors for next layer
		if i > 0 {
			grads, err := n.readStagingBuffer(layerCompute.stagingBuffer, int(layerCompute.outputSize))
			if err != nil {
				return fmt.Errorf("failed to read gradients for layer %d: %v", i+1, err)
			}
			errors = make([]float32, layerCompute.inputSize)
			for j, grad := range grads {
				if j < len(errors) {
					errors[j] = float32(grad)
				}
			}
		}
	}

	return nil
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
	n.gpu.optimized.initialized = false
	n.gpu.optimized = nil
}
