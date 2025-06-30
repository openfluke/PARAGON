// webgpu_wasm.go
//go:build js && wasm
// +build js,wasm

package paragon

import (
	"fmt"
	"syscall/js"

	"github.com/openfluke/webgpu/wgpu"
)

func initGPUForWASM() error {
	jsGlobal := js.Global()
	if !jsGlobal.Get("paragonWebGPU").Truthy() {
		return fmt.Errorf("JavaScript WebGPU context (paragonWebGPU) not found")
	}
	deviceJS := jsGlobal.Get("paragonWebGPU").Get("device")
	queueJS := jsGlobal.Get("paragonWebGPU").Get("queue")
	if !deviceJS.Truthy() {
		return fmt.Errorf("JavaScript WebGPU device not available")
	}
	if !queueJS.Truthy() {
		return fmt.Errorf("JavaScript WebGPU queue not available")
	}
	device := wgpu.NewDevice(deviceJS)
	ctx.device = &device
	ctx.queue = ctx.device.GetQueue()
	if ctx.device == nil {
		return fmt.Errorf("failed to initialize WebGPU device")
	}
	if ctx.queue == nil {
		return fmt.Errorf("failed to initialize WebGPU queue")
	}
	fmt.Println("Created WebGPU device successfully")
	return nil
}
