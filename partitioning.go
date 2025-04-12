package paragon

import "fmt"

func (n *Network) ForwardTagged(inputs [][]float64, numTags int, selectedTag int) {
	inputGrid := n.Layers[n.InputLayer]
	if len(inputs) != inputGrid.Height || len(inputs[0]) != inputGrid.Width {
		panic(fmt.Sprintf("input dimensions mismatch: expected %dx%d, got %dx%d", inputGrid.Height, inputGrid.Width, len(inputs), len(inputs[0])))
	}
	for y := 0; y < inputGrid.Height; y++ {
		for x := 0; x < inputGrid.Width; x++ {
			inputGrid.Neurons[y][x].Value = inputs[y][x]
		}
	}

	for l := 1; l < len(n.Layers); l++ {
		layer := n.Layers[l]
		if l == n.OutputLayer {
			// Compute all neurons in the output layer
			for y := 0; y < layer.Height; y++ {
				for x := 0; x < layer.Width; x++ {
					neuron := layer.Neurons[y][x]
					sum := neuron.Bias
					for _, conn := range neuron.Inputs {
						src := n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
						sum += src.Value * conn.Weight
					}
					//neuron.Value = applyActivation(sum, neuron.Activation)

					// *** ADDED FOR DIMENSIONAL NEURON ***
					if neuron.Dimension != nil {
						// We'll do the same "sub-network" forward logic that you have in Forward()
						subInLayer := neuron.Dimension.Layers[neuron.Dimension.InputLayer]
						inW := subInLayer.Width
						inH := subInLayer.Height
						totalIn := inW * inH

						if totalIn == 1 {
							// single scalar input
							subInput := [][]float64{{sum}}
							neuron.Dimension.Forward(subInput)
							subOut := neuron.Dimension.Layers[neuron.Dimension.OutputLayer].
								Neurons[0][0].Value
							neuron.Value = applyActivation(subOut, neuron.Activation)
						} else {
							// multiple inputs expected by the sub-network
							// for example, you might treat an entire row’s partial sums, etc.
							// The exact shape logic depends on how you want to feed it in.

							// Minimal example: if the sub-network is 1×N
							// you could pass [1][N] or [N][1], etc.
							// Here’s a quick example if sub-network is 1×1:
							panic("Sub-network has multi-input shape – handle it like in your Forward() code.")
						}
					} else {
						// normal activation
						neuron.Value = applyActivation(sum, neuron.Activation)
					}

					if n.Debug {
						fmt.Printf("Layer %d, Neuron (%d,%d): Value=%.4f\n", l, x, y, neuron.Value)
					}
				}
			}
		} else {
			// Compute only tagged section in hidden layers and zero others
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
						//neuron.Value = applyActivation(sum, neuron.Activation)

						// *** ADDED FOR DIMENSIONAL NEURON ***
						if neuron.Dimension != nil {
							// same idea as above
							subInLayer := neuron.Dimension.Layers[neuron.Dimension.InputLayer]
							inW := subInLayer.Width
							inH := subInLayer.Height
							totalIn := inW * inH

							if totalIn == 1 {
								subInput := [][]float64{{sum}}
								neuron.Dimension.Forward(subInput)
								subOut := neuron.Dimension.Layers[neuron.Dimension.OutputLayer].
									Neurons[0][0].Value
								neuron.Value = applyActivation(subOut, neuron.Activation)
							} else {
								// if sub-network expects more than 1 input, replicate the logic from your Forward()
								panic("Sub-network multi-input logic – replicate from your main Forward() code.")
							}
						} else {
							// normal activation
							neuron.Value = applyActivation(sum, neuron.Activation)
						}

					} else {
						layer.Neurons[y][x].Value = 0 // Zero out non-tagged neurons
					}
				}
			}
		}
	}

	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

func (n *Network) BackwardTagged(targets [][]float64, learningRate float64, numTags int, selectedTag int) {
	numLayers := len(n.Layers)
	errorTerms := make([][][]float64, numLayers)
	for l := range n.Layers {
		errorTerms[l] = make([][]float64, n.Layers[l].Height)
		for y := range errorTerms[l] {
			errorTerms[l][y] = make([]float64, n.Layers[l].Width)
		}
	}

	// Compute error for all output neurons
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			errorTerms[n.OutputLayer][y][x] = (targets[y][x] - neuron.Value) * activationDerivative(neuron.Value, neuron.Activation)
		}
	}

	for l := n.OutputLayer; l > 0; l-- {
		layer := n.Layers[l]
		prev := n.Layers[l-1]
		width := layer.Width
		startX := (width * selectedTag) / numTags
		endX := (width * (selectedTag + 1)) / numTags

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				if l == n.OutputLayer || (x >= startX && x < endX) { // Update all output neurons, only tagged hidden neurons
					neuron := layer.Neurons[y][x]
					err := errorTerms[l][y][x]
					neuron.Bias += learningRate * err
					for i, conn := range neuron.Inputs {
						srcNeuron := prev.Neurons[conn.SourceY][conn.SourceX]
						gradW := err * srcNeuron.Value
						if gradW > 5 {
							gradW = 5
						} else if gradW < -5 {
							gradW = -5
						}
						neuron.Inputs[i].Weight += learningRate * gradW
						if l-1 > 0 {
							// Only accumulate error terms for tagged section in hidden layers
							prevWidth := prev.Width
							prevStartX := (prevWidth * selectedTag) / numTags
							prevEndX := (prevWidth * (selectedTag + 1)) / numTags
							if conn.SourceX >= prevStartX && conn.SourceX < prevEndX {
								errorTerms[l-1][conn.SourceY][conn.SourceX] += err * conn.Weight
							}
						}
					}
				}
			}
		}

		// Apply activation derivative only to tagged section in hidden layers
		if l-1 > 0 {
			prevWidth := prev.Width
			prevStartX := (prevWidth * selectedTag) / numTags
			prevEndX := (prevWidth * (selectedTag + 1)) / numTags
			for y := 0; y < prev.Height; y++ {
				for x := prevStartX; x < prevEndX; x++ {
					errorTerms[l-1][y][x] *= activationDerivative(prev.Neurons[y][x].Value, prev.Neurons[y][x].Activation)
				}
			}
		}
	}
}
