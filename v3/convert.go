package paragon

import (
	"fmt"
	"math"
	"reflect"
)

// ConvertNetwork converts a network from one numeric type to another
func ConvertNetwork[T1, T2 Numeric](src *Network[T1]) (*Network[T2], error) {
	if src == nil {
		return nil, fmt.Errorf("source network is nil")
	}

	// Create destination network with same structure
	dst := &Network[T2]{
		TypeName:      reflect.TypeOf(*new(T2)).Name(),
		Layers:        make([]Grid[T2], len(src.Layers)),
		InputLayer:    src.InputLayer,
		OutputLayer:   src.OutputLayer,
		Debug:         src.Debug,
		Performance:   NewADHDPerformance(),
		Composite:     src.Composite,
		ReplayStats:   make(map[int][]int),
		WebGPUNative:  false, // Disable GPU for converted networks initially
		SCALE:         src.SCALE,
		GrowthHistory: src.GrowthHistory,
	}

	// Get scale factors for conversion
	srcScale := getScaleForType[T1]()
	dstScale := getScaleForType[T2]()

	// Convert each layer
	for i, srcLayer := range src.Layers {
		dstLayer := Grid[T2]{
			Width:            srcLayer.Width,
			Height:           srcLayer.Height,
			Neurons:          make([][]*Neuron[T2], srcLayer.Height),
			ReplayOffset:     srcLayer.ReplayOffset,
			ReplayPhase:      srcLayer.ReplayPhase,
			MaxReplay:        srcLayer.MaxReplay,
			ReplayEnabled:    srcLayer.ReplayEnabled,
			ReplayBudget:     srcLayer.ReplayBudget,
			ReplayGateFunc:   nil, // Functions can't be directly converted
			ReplayGateToReps: nil, // Functions can't be directly converted
			CachedOutputs:    make([]T2, len(srcLayer.CachedOutputs)),
		}

		// Convert cached outputs
		for j, val := range srcLayer.CachedOutputs {
			dstLayer.CachedOutputs[j] = convertValue[T1, T2](val, srcScale, dstScale)
		}

		// Convert neurons
		for y := 0; y < srcLayer.Height; y++ {
			dstLayer.Neurons[y] = make([]*Neuron[T2], srcLayer.Width)
			for x := 0; x < srcLayer.Width; x++ {
				srcNeuron := srcLayer.Neurons[y][x]
				dstNeuron := &Neuron[T2]{
					ID:         srcNeuron.ID,
					Value:      convertValue[T1, T2](srcNeuron.Value, srcScale, dstScale),
					Bias:       convertValue[T1, T2](srcNeuron.Bias, srcScale, dstScale),
					Activation: srcNeuron.Activation,
					Type:       srcNeuron.Type,
					Inputs:     make([]Connection[T2], len(srcNeuron.Inputs)),
					IsNew:      srcNeuron.IsNew,
					Dimension:  nil, // Sub-networks would need recursive conversion
					RevValue:   convertValue[T1, T2](srcNeuron.RevValue, srcScale, dstScale),
				}

				// Convert connections
				for j, srcConn := range srcNeuron.Inputs {
					dstNeuron.Inputs[j] = Connection[T2]{
						SourceLayer: srcConn.SourceLayer,
						SourceX:     srcConn.SourceX,
						SourceY:     srcConn.SourceY,
						Weight:      convertValue[T1, T2](srcConn.Weight, srcScale, dstScale),
					}
				}

				dstLayer.Neurons[y][x] = dstNeuron
			}
		}

		dst.Layers[i] = dstLayer
	}

	// Set up GPU type info
	dst.gpu.wgslType = getWGSLType[T2]()

	return dst, nil
}

// convertValue converts a single value from T1 to T2, handling scaling
func convertValue[T1, T2 Numeric](val T1, srcScale, dstScale int64) T2 {
	// Get type information
	srcKind := reflect.TypeOf(val).Kind()
	var dstZero T2
	dstKind := reflect.TypeOf(dstZero).Kind()

	// Convert to float64 as intermediate
	var floatVal float64

	// Source to float64
	switch srcKind {
	case reflect.Float32, reflect.Float64:
		floatVal = float64(any(val).(T1))
	default:
		// Integer types - unscale if needed
		if srcScale > 0 {
			floatVal = float64(any(val).(T1)) / float64(srcScale)
		} else {
			floatVal = float64(any(val).(T1))
		}
	}

	// Float64 to destination
	switch dstKind {
	case reflect.Float32:
		return T2(float32(floatVal))
	case reflect.Float64:
		return T2(floatVal)
	default:
		// Integer types - scale if needed
		if dstScale > 0 {
			scaledVal := floatVal * float64(dstScale)
			// Clamp to type bounds
			return clampToType[T2](scaledVal)
		} else {
			return clampToType[T2](floatVal)
		}
	}
}

// clampToType clamps a float64 value to the valid range of type T
func clampToType[T Numeric](val float64) T {
	var zero T
	kind := reflect.TypeOf(zero).Kind()

	// Round to nearest integer for integer types
	if kind != reflect.Float32 && kind != reflect.Float64 {
		val = math.Round(val)
	}

	// Clamp to type bounds using proper type-safe conversions
	switch kind {
	case reflect.Int8:
		if val < -128 {
			val = -128
		} else if val > 127 {
			val = 127
		}
		return T(any(int8(val)).(T))
	case reflect.Int16:
		if val < -32768 {
			val = -32768
		} else if val > 32767 {
			val = 32767
		}
		return T(any(int16(val)).(T))
	case reflect.Int32:
		if val < -2147483648 {
			val = -2147483648
		} else if val > 2147483647 {
			val = 2147483647
		}
		return T(any(int32(val)).(T))
	case reflect.Int64, reflect.Int:
		if val < -9223372036854775808 {
			val = -9223372036854775808
		} else if val > 9223372036854775807 {
			val = 9223372036854775807
		}
		return T(any(int64(val)).(T))
	case reflect.Uint8:
		if val < 0 {
			val = 0
		} else if val > 255 {
			val = 255
		}
		return T(any(uint8(val)).(T))
	case reflect.Uint16:
		if val < 0 {
			val = 0
		} else if val > 65535 {
			val = 65535
		}
		return T(any(uint16(val)).(T))
	case reflect.Uint32:
		if val < 0 {
			val = 0
		} else if val > 4294967295 {
			val = 4294967295
		}
		return T(any(uint32(val)).(T))
	case reflect.Uint64, reflect.Uint:
		if val < 0 {
			val = 0
		}
		// For uint64, we can't easily check the upper bound due to float64 precision
		return T(any(uint64(val)).(T))
	case reflect.Float32:
		return T(any(float32(val)).(T))
	case reflect.Float64:
		return T(any(val).(T))
	}

	return T(any(val).(T))
}

// BatchConvertNetworks converts a network to multiple numeric types
func BatchConvertNetworks[T Numeric](src *Network[T], types []string) (map[string]interface{}, error) {
	results := make(map[string]interface{})

	for _, typeName := range types {
		switch typeName {
		case "float32":
			if dst, err := ConvertNetwork[T, float32](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to float32: %w", err)
			}
		case "float64":
			if dst, err := ConvertNetwork[T, float64](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to float64: %w", err)
			}
		case "int8":
			if dst, err := ConvertNetwork[T, int8](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to int8: %w", err)
			}
		case "int16":
			if dst, err := ConvertNetwork[T, int16](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to int16: %w", err)
			}
		case "int32":
			if dst, err := ConvertNetwork[T, int32](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to int32: %w", err)
			}
		case "int64":
			if dst, err := ConvertNetwork[T, int64](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to int64: %w", err)
			}
		case "uint8":
			if dst, err := ConvertNetwork[T, uint8](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to uint8: %w", err)
			}
		case "uint16":
			if dst, err := ConvertNetwork[T, uint16](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to uint16: %w", err)
			}
		case "uint32":
			if dst, err := ConvertNetwork[T, uint32](src); err == nil {
				results[typeName] = dst
			} else {
				return nil, fmt.Errorf("failed to convert to uint32: %w", err)
			}
		default:
			return nil, fmt.Errorf("unsupported type: %s", typeName)
		}
	}

	return results, nil
}
