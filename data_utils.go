// data_utils.go
package paragon

import (
	"math/rand"
)

// SplitDataset takes inputs/targets and splits them into train/validation sets
// according to the given fraction (e.g., 0.8 for 80% train, 20% val).
func SplitDataset(inputs [][][]float64, targets [][][]float64, trainFrac float64) (
	trainIn [][][]float64, trainTarg [][][]float64,
	valIn [][][]float64, valTarg [][][]float64,
) {
	if len(inputs) != len(targets) {
		panic("inputs and targets length mismatch!")
	}
	n := len(inputs)
	trainSize := int(trainFrac * float64(n))

	perm := rand.Perm(n)
	trainIn = make([][][]float64, trainSize)
	trainTarg = make([][][]float64, trainSize)
	valIn = make([][][]float64, n-trainSize)
	valTarg = make([][][]float64, n-trainSize)

	for i, p := range perm {
		if i < trainSize {
			trainIn[i] = inputs[p]
			trainTarg[i] = targets[p]
		} else {
			valIn[i-trainSize] = inputs[p]
			valTarg[i-trainSize] = targets[p]
		}
	}
	return
}
