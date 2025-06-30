// webgpu_nonwasm.go
//go:build !js && !wasm
// +build !js,!wasm

package paragon

import "fmt"

func initGPUForWASM() error {
	return fmt.Errorf("WebGPU WASM initialization not supported in non-WASM builds")
}
