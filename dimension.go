package paragon

func (n *Network) SetLayerDimensionShared(
	layerIdx int,
	layerSizes []struct{ Width, Height int },
	activations []string,
	fullyConnected []bool,
) {
	// Create ONE sub-network
	sharedSubNet := NewNetwork(layerSizes, activations, fullyConnected)

	// Then assign the same pointer to each neuron in the chosen layer
	for y := 0; y < n.Layers[layerIdx].Height; y++ {
		for x := 0; x < n.Layers[layerIdx].Width; x++ {
			n.Layers[layerIdx].Neurons[y][x].Dimension = sharedSubNet
		}
	}
}

// SetLayerDimension instantiates a mini-network inside every neuron of a particular layer.
func (n *Network) SetLayerDimension(
	layerIdx int,
	layerSizes []struct{ Width, Height int },
	activations []string,
	fullyConnected []bool,
) {
	// We'll create a separate sub-network for each neuron in layerIdx.
	// You can decide if you want them to share the same sub-network instance
	// or each have their own. Here, each neuron gets its own copy.
	for y := 0; y < n.Layers[layerIdx].Height; y++ {
		for x := 0; x < n.Layers[layerIdx].Width; x++ {
			subNet := NewNetwork(layerSizes, activations, fullyConnected)
			// Example initialization, etc. (optionally set subNet.Debug = true, etc.)
			n.Layers[layerIdx].Neurons[y][x].Dimension = subNet
		}
	}
}
