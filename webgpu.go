package paragon

import (
	"fmt"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

func (n *Network[T]) ForwardLayerWebGPU(inputs, weights, biases []float32, outDim, inDim int) []float32 {
	device := n.device
	queue := n.queue
	pipeline := n.pipeline
	bindGroupLayout := n.bindGroupLayout

	if device == nil || queue == nil || pipeline == nil || bindGroupLayout == nil {
		panic("WebGPU resources not initialized. Call InitWebGPU first.")
	}

	// Create buffers for inputs, weights, biases, and outputs
	inBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "inputs",
		Contents: wgpu.ToBytes(inputs),
		Usage:    wgpu.BufferUsage_Storage,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create input buffer: %v", err))
	}
	defer inBuf.Release()

	wBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "weights",
		Contents: wgpu.ToBytes(weights),
		Usage:    wgpu.BufferUsage_Storage,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create weights buffer: %v", err))
	}
	defer wBuf.Release()

	bBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "biases",
		Contents: wgpu.ToBytes(biases),
		Usage:    wgpu.BufferUsage_Storage,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create biases buffer: %v", err))
	}
	defer bBuf.Release()

	sizeOutputs := uint64(outDim) * 4
	outGpuBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label:            "outputs",
		Size:             sizeOutputs,
		Usage:            wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc,
		MappedAtCreation: false,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create output buffer: %v", err))
	}
	defer outGpuBuf.Release()

	stagingBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label:            "staging",
		Size:             sizeOutputs,
		Usage:            wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
		MappedAtCreation: false,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create staging buffer: %v", err))
	}
	defer stagingBuf.Release()

	// Create bind group
	bindGroup, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inBuf, Size: wgpu.WholeSize},
			{Binding: 1, Buffer: wBuf, Size: wgpu.WholeSize},
			{Binding: 2, Buffer: bBuf, Size: wgpu.WholeSize},
			{Binding: 3, Buffer: outGpuBuf, Size: wgpu.WholeSize},
		},
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create bind group: %v", err))
	}
	defer bindGroup.Release()

	// Execute compute pass
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		panic(fmt.Sprintf("failed to create command encoder: %v", err))
	}
	defer encoder.Release()

	pass := encoder.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)

	// Calculate workgroups (2 elements per thread)
	const workgroupSize = 256
	const elementsPerThread = 2
	numThreads := (outDim + elementsPerThread - 1) / elementsPerThread
	numWorkgroups := (numThreads + workgroupSize - 1) / workgroupSize
	pass.DispatchWorkgroups(uint32(numWorkgroups), 1, 1)
	pass.End()

	// Copy results to staging buffer
	encoder.CopyBufferToBuffer(outGpuBuf, 0, stagingBuf, 0, sizeOutputs)
	cmdBuf, err := encoder.Finish(nil)
	if err != nil {
		panic(fmt.Sprintf("failed to finish command encoder: %v", err))
	}
	defer cmdBuf.Release()

	queue.Submit(cmdBuf)

	// Map staging buffer and retrieve results
	var status wgpu.BufferMapAsyncStatus
	err = stagingBuf.MapAsync(wgpu.MapMode_Read, 0, sizeOutputs, func(s wgpu.BufferMapAsyncStatus) {
		status = s
	})
	if err != nil {
		panic(fmt.Sprintf("failed to map staging buffer: %v", err))
	}
	defer stagingBuf.Unmap()

	device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatus_Success {
		panic(fmt.Sprintf("buffer mapping failed with status: %v", status))
	}

	outBuf := make([]float32, outDim)
	copy(outBuf, wgpu.FromBytes[float32](stagingBuf.GetMappedRange(0, uint(sizeOutputs))))

	return outBuf
}
