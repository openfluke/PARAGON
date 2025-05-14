package paragon

import "fmt"

func (n *Network[T]) ForwardTagged(inputs [][]float64, numTags int, selectedTag int) {
	inputGrid := n.Layers[n.InputLayer]

	if len(inputs) != inputGrid.Height || len(inputs[0]) != inputGrid.Width {
		panic(fmt.Sprintf("input dimensions mismatch: expected %dx%d, got %dx%d",
			inputGrid.Height, inputGrid.Width, len(inputs), len(inputs[0])))
	}

	// Inject inputs
	for y := 0; y < inputGrid.Height; y++ {
		for x := 0; x < inputGrid.Width; x++ {
			inputGrid.Neurons[y][x].Value = T(inputs[y][x])
		}
	}

	for l := 1; l < len(n.Layers); l++ {
		layer := n.Layers[l]

		if l == n.OutputLayer {
			// Full computation on output layer
			for y := 0; y < layer.Height; y++ {
				for x := 0; x < layer.Width; x++ {
					neuron := layer.Neurons[y][x]
					sum := neuron.Bias
					for _, conn := range neuron.Inputs {
						src := n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
						sum += src.Value * conn.Weight
					}

					// Dimensional neuron logic
					if neuron.Dimension != nil {
						subInLayer := neuron.Dimension.Layers[neuron.Dimension.InputLayer]
						totalIn := subInLayer.Width * subInLayer.Height

						if totalIn == 1 {
							subInput := [][]float64{{float64(any(sum).(T))}}
							neuron.Dimension.Forward(subInput)
							subOut := T(any(neuron.Dimension.
								Layers[neuron.Dimension.OutputLayer].
								Neurons[0][0].Value).(T))
							neuron.Value = ApplyActivationGeneric(subOut, neuron.Activation)
						} else {
							panic("Sub-network has multi-input shape – handle it like in your Forward() code.")
						}
					} else {
						neuron.Value = ApplyActivationGeneric(sum, neuron.Activation)
					}

					if n.Debug {
						fmt.Printf("Layer %d, Neuron (%d,%d): Value=%.4f\n", l, x, y,
							float64(any(neuron.Value).(T)))
					}
				}
			}

		} else {
			// Selectively compute neurons in the active tag block
			width := layer.Width
			startX := (width * selectedTag) / numTags
			endX := (width * (selectedTag + 1)) / numTags

			for y := 0; y < layer.Height; y++ {
				for x := 0; x < layer.Width; x++ {
					if x >= startX && x < endX {
						neuron := layer.Neurons[y][x]
						sum := neuron.Bias
						for _, conn := range neuron.Inputs {
							src := n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
							sum += src.Value * conn.Weight
						}

						if neuron.Dimension != nil {
							subInLayer := neuron.Dimension.Layers[neuron.Dimension.InputLayer]
							totalIn := subInLayer.Width * subInLayer.Height

							if totalIn == 1 {
								subInput := [][]float64{{float64(any(sum).(T))}}
								neuron.Dimension.Forward(subInput)
								subOut := T(any(neuron.Dimension.
									Layers[neuron.Dimension.OutputLayer].
									Neurons[0][0].Value).(T))
								neuron.Value = ApplyActivationGeneric(subOut, neuron.Activation)
							} else {
								panic("Sub-network multi-input logic – replicate from your main Forward() code.")
							}
						} else {
							neuron.Value = ApplyActivationGeneric(sum, neuron.Activation)
						}
					} else {
						layer.Neurons[y][x].Value = T(0)
					}
				}
			}
		}
	}

	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}
func (n *Network[T]) BackwardTagged(
	targets [][]float64,
	learningRate float64,
	numTags int,
	selectedTag int,
) {
	numLayers := len(n.Layers)
	errorTerms := make([][][]T, numLayers)

	for l := range n.Layers {
		layer := n.Layers[l]
		errorTerms[l] = make([][]T, layer.Height)
		for y := range errorTerms[l] {
			errorTerms[l][y] = make([]T, layer.Width)
		}
	}

	// Compute error for all output neurons
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			pred := float64(any(neuron.Value).(T))
			err := (targets[y][x] - pred) *
				float64(any(ActivationDerivativeGeneric(neuron.Value, neuron.Activation)).(T))
			errorTerms[n.OutputLayer][y][x] = T(err)
		}
	}

	// Backprop through tagged blocks
	for l := n.OutputLayer; l > 0; l-- {
		layer := n.Layers[l]
		prev := n.Layers[l-1]

		width := layer.Width
		startX := (width * selectedTag) / numTags
		endX := (width * (selectedTag + 1)) / numTags

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				// Train all output neurons, but only tagged hidden neurons
				if l == n.OutputLayer || (x >= startX && x < endX) {
					neuron := layer.Neurons[y][x]
					err := errorTerms[l][y][x]
					neuron.Bias += T(learningRate) * err

					for i, conn := range neuron.Inputs {
						srcNeuron := prev.Neurons[conn.SourceY][conn.SourceX]
						grad := err * srcNeuron.Value

						// Clip gradient
						var clippedGrad T
						switch any(clippedGrad).(type) {
						case float32, float64:
							fg := float64(any(grad).(T))
							if fg > 5 {
								fg = 5
							} else if fg < -5 {
								fg = -5
							}
							clippedGrad = T(fg)
						default:
							clippedGrad = grad // no clip for integers
						}

						neuron.Inputs[i].Weight += T(learningRate) * clippedGrad

						// Accumulate tagged error in previous layer
						if l-1 > 0 {
							prevWidth := prev.Width
							prevStart := (prevWidth * selectedTag) / numTags
							prevEnd := (prevWidth * (selectedTag + 1)) / numTags
							if conn.SourceX >= prevStart && conn.SourceX < prevEnd {
								errorTerms[l-1][conn.SourceY][conn.SourceX] += err * conn.Weight
							}
						}
					}
				}
			}
		}

		// Chain rule: apply derivative to tagged block of previous layer
		if l-1 > 0 {
			prevWidth := n.Layers[l-1].Width
			start := (prevWidth * selectedTag) / numTags
			end := (prevWidth * (selectedTag + 1)) / numTags
			for y := 0; y < n.Layers[l-1].Height; y++ {
				for x := start; x < end; x++ {
					neuron := n.Layers[l-1].Neurons[y][x]
					errorTerms[l-1][y][x] *= ActivationDerivativeGeneric(neuron.Value, neuron.Activation)
				}
			}
		}
	}
}
