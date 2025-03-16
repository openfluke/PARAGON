// metrics.go
package paragon

// ComputeAccuracy calculates the classification accuracy for a dataset.
func ComputeAccuracy(nn *Network, inputs [][][]float64, targets [][][]float64) float64 {
	if len(inputs) == 0 {
		return 0
	}
	correct := 0
	for i := range inputs {
		nn.Forward(inputs[i])

		// Suppose the output layer is 1D or 2D with (Height=1), and we have a one-hot target.
		// This is typical for classification, but can be extended as needed.
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}

		pred := argMax(outputValues)
		label := argMax(targets[i][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// EvaluateWithADHD runs the ADHD evaluation on a classification dataset.
func EvaluateWithADHD(nn *Network, inputs [][][]float64, targets [][][]float64) {
	if len(inputs) == 0 {
		return
	}

	expectedOutputs := make([]float64, len(inputs))
	actualOutputs := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}
		pred := argMax(outputValues)
		label := argMax(targets[i][0])

		expectedOutputs[i] = float64(label)
		actualOutputs[i] = float64(pred)
	}

	nn.EvaluateModel(expectedOutputs, actualOutputs) // The ADHD logic from adhd.go
}

// argMax finds the index of the largest value in a 1D slice.
func argMax(arr []float64) int {
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
