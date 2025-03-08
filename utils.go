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
