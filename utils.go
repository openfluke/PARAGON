package paragon

import "math"

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
