package paragon

// PropagateProxyError adjusts weights and biases using a proxy input signal.
// The signal decays by proxyDecay per layer. This is an alternative to gradient-based backprop.
func (n *Network) PropagateProxyError(input [][]float64, errorSignal, lr, maxUpdate, damping, proxyDecay float64) {
	var proxySignal float64
	count := 0
	for _, row := range input {
		for _, v := range row {
			proxySignal += v
			count++
		}
	}
	if count > 0 {
		proxySignal /= float64(count)
	}

	for layerIndex := n.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &n.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				adj := lr * errorSignal * damping

				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				neuron.Bias += adj
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += adj * proxySignal
				}
			}
		}
		proxySignal *= proxyDecay
	}
}

// ReverseInferFromOutput starts with an output layer state and infers an approximate input.
func (n *Network) ReverseInferFromOutput(output [][]float64) [][]float64 {
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			outputLayer.Neurons[y][x].Value = output[y][x]
		}
	}

	for layerIndex := n.OutputLayer; layerIndex > n.InputLayer; layerIndex-- {
		currLayer := n.Layers[layerIndex]
		prevLayer := n.Layers[layerIndex-1]

		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				var sum float64
				var count int

				for cy := 0; cy < currLayer.Height; cy++ {
					for cx := 0; cx < currLayer.Width; cx++ {
						neuron := currLayer.Neurons[cy][cx]
						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == layerIndex-1 &&
								conn.SourceY == y &&
								conn.SourceX == x &&
								conn.Weight != 0 {

								approx := (neuron.Value - neuron.Bias) / conn.Weight
								sum += approx
								count++
							}
						}
					}
				}

				if count > 0 {
					prevLayer.Neurons[y][x].Value = sum / float64(count)
				}
			}
		}
	}

	inputLayer := n.Layers[n.InputLayer]
	inferred := make([][]float64, inputLayer.Height)
	for y := 0; y < inputLayer.Height; y++ {
		inferred[y] = make([]float64, inputLayer.Width)
		for x := 0; x < inputLayer.Width; x++ {
			inferred[y][x] = inputLayer.Neurons[y][x].Value
		}
	}
	return inferred
}

func (n *Network) TuneWithReverseAttribution(actualInput, targetOutput [][]float64, stepSize float64) {
	// Step 1: Run reverse inference from output
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Compute total number of biases + weights
	var totalParams int
	for layerIndex := 1; layerIndex <= n.OutputLayer; layerIndex++ {
		layer := n.Layers[layerIndex]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				totalParams++ // bias
				totalParams += len(neuron.Inputs)
			}
		}
	}

	if totalParams == 0 {
		return
	}

	// Step 3: Adjust each parameter proportionally to Δinput
	for layerIndex := 1; layerIndex <= n.OutputLayer; layerIndex++ {
		layer := n.Layers[layerIndex]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Accumulate delta from input discrepancy
				var deltaSum float64
				for iy := 0; iy < len(actualInput) && iy < len(reconstructed); iy++ {
					for ix := 0; ix < len(actualInput[0]) && ix < len(reconstructed[0]); ix++ {
						delta := actualInput[iy][ix] - reconstructed[iy][ix]
						deltaSum += delta
					}
				}
				// Scale down delta to each parameter's share
				perParamAdj := (stepSize * deltaSum) / float64(totalParams)

				// Bias update
				neuron.Bias += perParamAdj

				// Weight updates
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += perParamAdj
				}
			}
		}
	}
}

func (n *Network) SetBiasWeightFromReverseAttribution(actualInput, targetOutput [][]float64, scale float64) {
	// Step 1: Reverse infer the input
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Compute total delta between actual and reconstructed input
	var totalDelta float64
	for y := range actualInput {
		for x := range actualInput[y] {
			totalDelta += actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 3: Count total parameters (weights + biases)
	var totalParams int
	for l := 1; l <= n.OutputLayer; l++ {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				neuron := n.Layers[l].Neurons[y][x] // ✅ single pointer
				totalParams++                       // bias
				totalParams += len(neuron.Inputs)   // weights
			}
		}
	}
	if totalParams == 0 {
		return
	}

	// Step 4: Calculate the value to set all weights and biases to
	valuePerParam := (scale * totalDelta) / float64(totalParams)

	// Step 5: Overwrite all parameters with this value
	for l := 1; l <= n.OutputLayer; l++ {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				neuron := n.Layers[l].Neurons[y][x] // ✅ correct pointer
				neuron.Bias = valuePerParam
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight = valuePerParam
				}
			}
		}
	}
}

func (n *Network) SetBiasWeightFromReverseAttributionPercent(actualInput, targetOutput [][]float64, percent float64) {
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 1: Compute per-pixel delta
	deltaMap := make([][]float64, len(actualInput))
	for y := range actualInput {
		deltaMap[y] = make([]float64, len(actualInput[y]))
		for x := range actualInput[y] {
			deltaMap[y][x] = actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 2: Walk through the network, adjusting each weight and bias partially
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Bias: move partially toward delta-sum average
				var influence float64
				for dy := range deltaMap {
					for dx := range deltaMap[dy] {
						influence += deltaMap[dy][dx]
					}
				}
				avgInfluence := influence / float64(len(deltaMap)*len(deltaMap[0]))
				neuron.Bias = (1.0-percent)*neuron.Bias + percent*avgInfluence

				// Weights: each weight gets same directional nudge (for now)
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight = (1.0-percent)*neuron.Inputs[i].Weight + percent*avgInfluence
				}
			}
		}
	}
}

func (n *Network) PropagateBidirectionalConstraint(actualInput, targetOutput [][]float64, percent float64, decay float64) {
	// Step 1: Infer input from the output
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Build delta map of where reality differs from hallucination
	deltaMap := make([][]float64, len(actualInput))
	for y := range actualInput {
		deltaMap[y] = make([]float64, len(actualInput[y]))
		for x := range actualInput[y] {
			deltaMap[y][x] = actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 3: Walk backwards through the network applying correction
	strength := percent

	for layerIndex := n.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := n.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// 3a. Use avg delta as estimated correction signal
				var influence float64
				for dy := range deltaMap {
					for dx := range deltaMap[dy] {
						influence += deltaMap[dy][dx]
					}
				}
				avgInfluence := influence / float64(len(deltaMap)*len(deltaMap[0]))

				// 3b. Update bias using this backward influence
				neuron.Bias = (1.0-strength)*neuron.Bias + strength*avgInfluence

				// 3c. Update weights similarly (assumes shared influence)
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight = (1.0-strength)*neuron.Inputs[i].Weight + strength*avgInfluence
				}
			}
		}

		// 3d. Decay the correction signal per layer
		strength *= decay
	}
}

func (n *Network) PropagateSandwichConstraint(actualInput, targetOutput [][]float64, lr, decay float64) {
	n.Forward(actualInput)
	n.ReverseInferFromOutputWithTrace(targetOutput)

	strength := lr

	for layerIndex := 1; layerIndex <= n.OutputLayer; layerIndex++ {
		layer := n.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				delta := neuron.RevValue - neuron.Value

				// Update bias
				neuron.Bias += strength * delta

				// Update weights using delta and input value
				for i := range neuron.Inputs {
					in := neuron.Inputs[i]
					inputVal := n.Layers[in.SourceLayer].Neurons[in.SourceY][in.SourceX].Value
					neuron.Inputs[i].Weight += strength * delta * inputVal
				}
			}
		}
		strength *= decay
	}
}

// ReverseInferFromOutputWithTrace does reverse inference from output,
// but stores per-neuron RevValue for internal training use.
func (n *Network) ReverseInferFromOutputWithTrace(output [][]float64) {
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			neuron.Value = output[y][x]
			neuron.RevValue = output[y][x]
		}
	}

	for layerIndex := n.OutputLayer; layerIndex > n.InputLayer; layerIndex-- {
		currLayer := n.Layers[layerIndex]
		prevLayer := n.Layers[layerIndex-1]

		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				var sum float64
				var count int

				for cy := 0; cy < currLayer.Height; cy++ {
					for cx := 0; cx < currLayer.Width; cx++ {
						neuron := currLayer.Neurons[cy][cx]
						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == layerIndex-1 &&
								conn.SourceY == y &&
								conn.SourceX == x &&
								conn.Weight != 0 {

								approx := (neuron.RevValue - neuron.Bias) / conn.Weight
								sum += approx
								count++
							}
						}
					}
				}

				if count > 0 {
					rev := sum / float64(count)
					prevLayer.Neurons[y][x].RevValue = rev
				}
			}
		}
	}
}
