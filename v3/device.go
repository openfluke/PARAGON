package paragon

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// GetAllGPUInfo retrieves information about all available GPUs, excluding CPU-based adapters,
// and returns it as a slice of maps with string values for adapter properties and limits.
// All WebGPU resources are automatically cleaned up.
func GetAllGPUInfo() ([]map[string]string, error) {
    // Create WebGPU instance
    instance := wgpu.CreateInstance(nil)
    if instance == nil {
        return nil, fmt.Errorf("failed to create WebGPU instance")
    }
    defer instance.Release()

    // Enumerate all adapters
    adapters := instance.EnumerateAdapters(nil)
    if len(adapters) == 0 {
        return nil, fmt.Errorf("no GPU adapters found")
    }

    var gpuInfo []map[string]string
    for i, adapter := range adapters {
        // Get adapter properties
        props := adapter.GetInfo()

        // Skip CPU-based adapters
        if fmt.Sprintf("%v", props.AdapterType) == "cpu" {
            adapter.Release()
            continue
        }

        // Create device
        device, err := adapter.RequestDevice(nil)
        if err != nil {
            adapter.Release()
            continue
        }

        // Get device limits
        limits := adapter.GetLimits()

        // Build GPU info map with string values
        info := map[string]string{
            "index":                        fmt.Sprintf("%d", i),
            "name":                         props.Name,
            "driverDescription":            props.DriverDescription,
            "adapterType":                  fmt.Sprintf("%v", props.AdapterType),
            "vendorId":                     fmt.Sprintf("0x%X", props.VendorId),
            "vendorName":                   props.VendorName,
            "architecture":                 props.Architecture,
            "deviceId":                     fmt.Sprintf("0x%X", props.DeviceId),
            "backendType":                  fmt.Sprintf("%v", props.BackendType),
            "maxComputeInvocations":        fmt.Sprintf("%d", limits.Limits.MaxComputeInvocationsPerWorkgroup),
            "maxComputeWorkgroupSizeX":     fmt.Sprintf("%d", limits.Limits.MaxComputeWorkgroupSizeX),
            "maxComputeWorkgroupSizeY":     fmt.Sprintf("%d", limits.Limits.MaxComputeWorkgroupSizeY),
            "maxComputeWorkgroupSizeZ":     fmt.Sprintf("%d", limits.Limits.MaxComputeWorkgroupSizeZ),
            "maxBufferSizeMB":              fmt.Sprintf("%.2f", float64(limits.Limits.MaxBufferSize)/1024/1024),
            "maxStorageBufferSizeMB":       fmt.Sprintf("%.2f", float64(limits.Limits.MaxStorageBufferBindingSize)/1024/1024),
            "maxUniformBufferSizeMB":       fmt.Sprintf("%.2f", float64(limits.Limits.MaxUniformBufferBindingSize)/1024/1024),
            "maxComputeWorkgroupStorageKB": fmt.Sprintf("%.2f", float64(limits.Limits.MaxComputeWorkgroupStorageSize)/1024),
        }

        gpuInfo = append(gpuInfo, info)

        // Cleanup
        device.Release()
        adapter.Release()
    }

    if len(gpuInfo) == 0 {
        return nil, fmt.Errorf("no valid GPU information retrieved")
    }

    return gpuInfo, nil
}

func (n *Network[T]) GetAllGPUInfo() ([]map[string]string, error) {
    return GetAllGPUInfo()
}