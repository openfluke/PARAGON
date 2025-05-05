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
