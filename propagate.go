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
