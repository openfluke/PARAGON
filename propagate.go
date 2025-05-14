package paragon

// PropagateProxyError adjusts weights and biases using a proxy input signal.
// The signal decays by proxyDecay per layer. This is an alternative to gradient-based backprop.
func (n *Network[T]) PropagateProxyError(
	input [][]float64,
	errorSignal, lr, maxUpdate, damping, proxyDecay float64,
) {
	// Compute average proxy signal from input
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

	// Backward proxy propagation
	for layerIndex := n.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &n.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				rawAdj := errorSignal * lr * damping

				// Clip to ±maxUpdate
				if rawAdj > maxUpdate {
					rawAdj = maxUpdate
				} else if rawAdj < -maxUpdate {
					rawAdj = -maxUpdate
				}

				adj := T(rawAdj)
				neuron.Bias += adj

				for i := range neuron.Inputs {
					delta := T(rawAdj * proxySignal)
					neuron.Inputs[i].Weight += delta
				}
			}
		}
		proxySignal *= proxyDecay
	}
}

// ReverseInferFromOutput starts with an output layer state and infers an approximate input.
func (n *Network[T]) ReverseInferFromOutput(output [][]float64) [][]float64 {
	outputLayer := n.Layers[n.OutputLayer]

	// Inject desired output into output layer
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			outputLayer.Neurons[y][x].Value = T(output[y][x])
		}
	}

	// Backward inference pass
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
						nVal := float64(any(neuron.Value).(T))
						nBias := float64(any(neuron.Bias).(T))

						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == layerIndex-1 &&
								conn.SourceY == y &&
								conn.SourceX == x &&
								conn.Weight != 0 {

								w := float64(any(conn.Weight).(T))
								approx := (nVal - nBias) / w
								sum += approx
								count++
							}
						}
					}
				}

				if count > 0 {
					prevLayer.Neurons[y][x].Value = T(sum / float64(count))
				}
			}
		}
	}

	// Extract inferred input as [][]float64
	inputLayer := n.Layers[n.InputLayer]
	inferred := make([][]float64, inputLayer.Height)
	for y := 0; y < inputLayer.Height; y++ {
		inferred[y] = make([]float64, inputLayer.Width)
		for x := 0; x < inputLayer.Width; x++ {
			inferred[y][x] = float64(any(inputLayer.Neurons[y][x].Value).(T))
		}
	}

	return inferred
}

func (n *Network[T]) TuneWithReverseAttribution(
	actualInput, targetOutput [][]float64,
	stepSize float64,
) {
	// Step 1: Reverse inference from target output
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Count total parameters (bias + weights)
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

	// Step 3: Accumulate Δinput
	var deltaSum float64
	for iy := 0; iy < len(actualInput) && iy < len(reconstructed); iy++ {
		for ix := 0; ix < len(actualInput[0]) && ix < len(reconstructed[0]); ix++ {
			delta := actualInput[iy][ix] - reconstructed[iy][ix]
			deltaSum += delta
		}
	}

	// Compute shared per-parameter adjustment
	perParamAdj := (stepSize * deltaSum) / float64(totalParams)
	adj := T(perParamAdj)

	// Step 4: Apply adjustment to all parameters
	for layerIndex := 1; layerIndex <= n.OutputLayer; layerIndex++ {
		layer := n.Layers[layerIndex]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Update bias
				neuron.Bias += adj

				// Update weights
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += adj
				}
			}
		}
	}
}
func (n *Network[T]) SetBiasWeightFromReverseAttribution(
	actualInput, targetOutput [][]float64,
	scale float64,
) {
	// Step 1: Reverse infer the input from the target output
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Compute total delta between actual and reconstructed input
	var totalDelta float64
	for y := range actualInput {
		for x := range actualInput[y] {
			totalDelta += actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 3: Count total parameters (biases + weights)
	var totalParams int
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				totalParams++                     // bias
				totalParams += len(neuron.Inputs) // weights
			}
		}
	}
	if totalParams == 0 {
		return
	}

	// Step 4: Compute shared attribution-driven parameter value
	valuePerParam := (scale * totalDelta) / float64(totalParams)
	v := T(valuePerParam)

	// Step 5: Set all biases and weights to the same value
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				neuron.Bias = v
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight = v
				}
			}
		}
	}
}

func (n *Network[T]) SetBiasWeightFromReverseAttributionPercent(
	actualInput, targetOutput [][]float64,
	percent float64,
) {
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 1: Compute per-pixel delta
	deltaMap := make([][]float64, len(actualInput))
	for y := range actualInput {
		deltaMap[y] = make([]float64, len(actualInput[y]))
		for x := range actualInput[y] {
			deltaMap[y][x] = actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 2: Calculate average influence across the entire input
	var influence float64
	for y := range deltaMap {
		for x := range deltaMap[y] {
			influence += deltaMap[y][x]
		}
	}
	numPixels := len(deltaMap) * len(deltaMap[0])
	if numPixels == 0 {
		return
	}
	avgInfluence := influence / float64(numPixels)

	// Step 3: Adjust all biases and weights partially toward the attribution average
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Bias: interpolate between old and new
				oldBias := float64(any(neuron.Bias).(T))
				newBias := (1.0-percent)*oldBias + percent*avgInfluence
				neuron.Bias = T(newBias)

				// Weights: same interpolated influence
				for i := range neuron.Inputs {
					oldWeight := float64(any(neuron.Inputs[i].Weight).(T))
					newWeight := (1.0-percent)*oldWeight + percent*avgInfluence
					neuron.Inputs[i].Weight = T(newWeight)
				}
			}
		}
	}
}

func (n *Network[T]) PropagateBidirectionalConstraint(
	actualInput, targetOutput [][]float64,
	percent float64,
	decay float64,
) {
	// Step 1: Reverse inference to reconstruct input
	reconstructed := n.ReverseInferFromOutput(targetOutput)

	// Step 2: Build delta map
	deltaMap := make([][]float64, len(actualInput))
	for y := range actualInput {
		deltaMap[y] = make([]float64, len(actualInput[y]))
		for x := range actualInput[y] {
			deltaMap[y][x] = actualInput[y][x] - reconstructed[y][x]
		}
	}

	// Step 3: Apply corrections backward through the network
	strength := percent

	for layerIndex := n.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := n.Layers[layerIndex]

		// Compute average influence from the deltaMap
		var influence float64
		for y := range deltaMap {
			for x := range deltaMap[y] {
				influence += deltaMap[y][x]
			}
		}
		numPixels := len(deltaMap) * len(deltaMap[0])
		if numPixels == 0 {
			return
		}
		avgInfluence := influence / float64(numPixels)

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Bias update
				oldBias := float64(any(neuron.Bias).(T))
				newBias := (1.0-strength)*oldBias + strength*avgInfluence
				neuron.Bias = T(newBias)

				// Weight updates
				for i := range neuron.Inputs {
					oldWeight := float64(any(neuron.Inputs[i].Weight).(T))
					newWeight := (1.0-strength)*oldWeight + strength*avgInfluence
					neuron.Inputs[i].Weight = T(newWeight)
				}
			}
		}

		// Decay strength after each layer
		strength *= decay
	}
}

func (n *Network[T]) PropagateSandwichConstraint(
	actualInput, targetOutput [][]float64,
	lr, decay float64,
) {
	// Forward pass with true input
	n.Forward(actualInput)

	// Reverse pass that populates RevValue
	n.ReverseInferFromOutputWithTrace(targetOutput)

	strength := lr

	for layerIndex := 1; layerIndex <= n.OutputLayer; layerIndex++ {
		layer := n.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				forwardVal := float64(any(neuron.Value).(T))
				reverseVal := float64(any(neuron.RevValue).(T))
				delta := reverseVal - forwardVal

				// Update bias
				bAdj := strength * delta
				neuron.Bias += T(bAdj)

				// Update weights
				for i := range neuron.Inputs {
					in := neuron.Inputs[i]
					inputVal := float64(any(n.Layers[in.SourceLayer].Neurons[in.SourceY][in.SourceX].Value).(T))
					grad := strength * delta * inputVal
					neuron.Inputs[i].Weight += T(grad)
				}
			}
		}
		strength *= decay
	}
}

// ReverseInferFromOutputWithTrace does reverse inference from output,
// but stores per-neuron RevValue for internal training use.
func (n *Network[T]) ReverseInferFromOutputWithTrace(output [][]float64) {
	outputLayer := n.Layers[n.OutputLayer]

	// Inject desired output values and set initial RevValue
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			val := T(output[y][x])
			outputLayer.Neurons[y][x].Value = val
			outputLayer.Neurons[y][x].RevValue = val
		}
	}

	// Propagate reverse approximation backward
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
						revVal := float64(any(neuron.RevValue).(T))
						bias := float64(any(neuron.Bias).(T))

						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == layerIndex-1 &&
								conn.SourceY == y &&
								conn.SourceX == x &&
								conn.Weight != 0 {

								weight := float64(any(conn.Weight).(T))
								approx := (revVal - bias) / weight
								sum += approx
								count++
							}
						}
					}
				}

				if count > 0 {
					avg := sum / float64(count)
					prevLayer.Neurons[y][x].RevValue = T(avg)
				}
			}
		}
	}
}
func (n *Network[T]) InferInputFromOutput(
	targetOutput [][]float64,
	steps int,
	lr float64,
) [][]float64 {
	inputLayer := n.Layers[n.InputLayer]
	height := inputLayer.Height
	width := inputLayer.Width

	// Step 0: Initialize guess with midpoint input
	input := make([][]float64, height)
	for y := range input {
		input[y] = make([]float64, width)
		for x := range input[y] {
			input[y][x] = 0.5 // Neutral gray
		}
	}

	for step := 0; step < steps; step++ {
		// Step 1: Forward with current input guess
		n.Forward(input)

		// Step 2: Allocate error tensor
		errorTerms := make([][][]T, len(n.Layers))
		for l := range n.Layers {
			layer := n.Layers[l]
			errorTerms[l] = make([][]T, layer.Height)
			for y := range errorTerms[l] {
				errorTerms[l][y] = make([]T, layer.Width)
			}
		}

		// Step 3: Output error
		outputLayer := n.Layers[n.OutputLayer]
		for y := 0; y < outputLayer.Height; y++ {
			for x := 0; x < outputLayer.Width; x++ {
				neuron := outputLayer.Neurons[y][x]
				pred := float64(any(neuron.Value).(T))
				targ := targetOutput[y][x]
				err := (targ - pred) *
					float64(any(ActivationDerivativeGeneric(neuron.Value, neuron.Activation)).(T))
				errorTerms[n.OutputLayer][y][x] = T(err)
			}
		}

		// Step 4: Backpropagate error
		for l := n.OutputLayer; l > 0; l-- {
			curr := n.Layers[l]
			//prev := n.Layers[l-1]

			for y := 0; y < curr.Height; y++ {
				for x := 0; x < curr.Width; x++ {
					err := errorTerms[l][y][x]
					for _, conn := range curr.Neurons[y][x].Inputs {
						errorTerms[l-1][conn.SourceY][conn.SourceX] += err * conn.Weight
					}
				}
			}

			// Step 5: Update input at bottom layer
			if l-1 == n.InputLayer {
				for y := 0; y < height; y++ {
					for x := 0; x < width; x++ {
						neuron := inputLayer.Neurons[y][x]
						grad := float64(any(errorTerms[n.InputLayer][y][x]).(T)) *
							float64(any(ActivationDerivativeGeneric(neuron.Value, neuron.Activation)).(T))
						input[y][x] += lr * grad

						// Clamp to valid input range [0, 1]
						if input[y][x] < 0 {
							input[y][x] = 0
						} else if input[y][x] > 1 {
							input[y][x] = 1
						}
					}
				}
				break
			}
		}
	}

	return input
}

func (n *Network[T]) BackwardFromOutput(targetOutput [][]float64) [][]float64 {
	// Step 1: Set output values directly
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			val := T(targetOutput[y][x])
			outputLayer.Neurons[y][x].Value = val
		}
	}

	// Step 2: Propagate backward approximation
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
						nVal := float64(any(neuron.Value).(T))
						nBias := float64(any(neuron.Bias).(T))

						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == layerIndex-1 &&
								conn.SourceY == y &&
								conn.SourceX == x &&
								conn.Weight != 0 {

								weight := float64(any(conn.Weight).(T))
								approx := (nVal - nBias) / weight

								// Inverse activation approximation
								switch neuron.Activation {
								case "leaky_relu":
									if approx < 0 {
										approx *= 100.0 // Invert 0.01 slope
									}
									// Add more inverse rules here if needed
								}

								sum += approx
								count++
							}
						}
					}
				}

				if count > 0 {
					prevLayer.Neurons[y][x].Value = T(sum / float64(count))
				}
			}
		}
	}

	// Step 3: Extract inferred input
	inputLayer := n.Layers[n.InputLayer]
	inferred := make([][]float64, inputLayer.Height)
	for y := 0; y < inputLayer.Height; y++ {
		inferred[y] = make([]float64, inputLayer.Width)
		for x := 0; x < inputLayer.Width; x++ {
			inferred[y][x] = float64(any(inputLayer.Neurons[y][x].Value).(T))
		}
	}

	return inferred
}
