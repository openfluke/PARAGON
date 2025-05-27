package paragon

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"reflect"
)

// Softmax computes the softmax of a slice
func Softmax(inputs []float64) []float64 {
	maxVal := inputs[0]
	for _, v := range inputs {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expInputs := make([]float64, len(inputs))
	for i, v := range inputs {
		expInputs[i] = math.Exp(v - maxVal)
		expSum += expInputs[i]
	}
	for i := range expInputs {
		expInputs[i] /= expSum
	}
	return expInputs
}

// ArgMax returns the index of the maximum value in the slice.
// If the slice is empty, it returns -1.
func ArgMax(arr []float64) int {
	if len(arr) == 0 {
		return -1
	}
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

// FileExists checks if a file exists at the given path and returns true if it does, false otherwise.
func FileExists(filename string) bool {
	_, err := os.Stat(filename)
	if err == nil {
		return true // File exists
	}
	if os.IsNotExist(err) {
		return false // File does not exist
	}
	// If there's another error (e.g., permission denied), assume the file doesn't exist for safety
	return false
}

// ReadCSV reads a CSV file without headers and returns its contents as a [][]string array.
// Each row in the CSV file becomes a slice of strings in the returned array.
func ReadCSV(filename string) ([][]string, error) {
	// Check if the file exists
	if !FileExists(filename) {
		return nil, fmt.Errorf("file %s does not exist", filename)
	}

	// Open the CSV file
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file %s: %v", filename, err)
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	// Read all rows from the CSV file
	data, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV file %s: %v", filename, err)
	}

	// If the file is empty, return an empty slice
	if len(data) == 0 {
		return [][]string{}, nil
	}

	return data, nil
}

// PadInputToFullSize pads a partial input slice with a given value so it matches the network's full input size.
// This is useful when only some inputs are available and you want to fill the rest with a default.
func (n *Network[T]) PadInputToFullSize(partial []float64, fill float64) []float64 {
	inputLayer := n.Layers[n.InputLayer]
	totalSize := inputLayer.Width * inputLayer.Height

	result := make([]float64, totalSize)
	copy(result, partial)

	for i := len(partial); i < totalSize; i++ {
		result[i] = fill
	}

	return result
}

func CastFloat64SliceToT[T Numeric](in []float64) []T {
	out := make([]T, len(in))
	for i := range in {
		out[i] = T(in[i])
	}
	return out
}

func calculateMaxVal[T any](numInputs int) uint64 {
	var weight T
	var typeMax uint64
	switch reflect.TypeOf(weight).Kind() {
	case reflect.Uint:
		typeMax = uint64(^uint(0)) // System-dependent max uint
	case reflect.Uint8:
		typeMax = math.MaxUint8
	case reflect.Uint64:
		typeMax = math.MaxUint64
	default:
		typeMax = math.MaxUint64 // Default case
	}

	maxValFloat := float64(typeMax) / math.Sqrt(float64(numInputs))
	if maxValFloat > float64(typeMax) {
		maxValFloat = float64(typeMax)
	}
	maxVal := uint64(maxValFloat)
	if maxVal < 1 {
		maxVal = 1
	}
	return maxVal
}

func softmax32(in []float32) []float32 {
	maxVal := in[0]
	for _, v := range in {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := float32(0.0)
	for _, v := range in {
		expSum += float32(math.Exp(float64(v - maxVal)))
	}
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(math.Exp(float64(v-maxVal))) / expSum
	}
	return out
}

func applyActivationFloat32(value float32, activation string) float32 {
	switch activation {
	case "relu":
		if value > 0 {
			return value
		}
		return 0
	case "sigmoid":
		return 1 / (1 + float32(math.Exp(-float64(value))))
	case "tanh":
		return float32(math.Tanh(float64(value)))
	default:
		return value
	}
}

// flatten2DF64 turns [][]float64 into one row-major []float64.
func flatten2DF64(src [][]float64) []float64 {
	if len(src) == 0 {
		return nil
	}
	out := make([]float64, 0, len(src)*len(src[0]))
	for _, r := range src {
		out = append(out, r...)
	}
	return out
}
