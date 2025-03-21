package paragon

import (
	"math"
	"math/rand"
)

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

// SetLayerDimensionOptions defines sub-network setup options
type SetLayerDimensionOptions struct {
	Shared     bool   // If true, all neurons share the same sub-network instance
	InitMethod string // "xavier" or "he" for weight initialization
}

// SetLayerDimension initializes sub-networks with options
func (n *Network) SetLayerDimension(
	layerIdx int,
	layerSizes []struct{ Width, Height int },
	activations []string,
	fullyConnected []bool,
	opts SetLayerDimensionOptions,
) {
	if layerIdx < 0 || layerIdx >= len(n.Layers) {
		return
	}

	var sharedSubNet *Network
	if opts.Shared {
		sharedSubNet = NewNetwork(layerSizes, activations, fullyConnected)
		initializeWeights(sharedSubNet, opts.InitMethod)
	}

	for y := 0; y < n.Layers[layerIdx].Height; y++ {
		for x := 0; x < n.Layers[layerIdx].Width; x++ {
			if opts.Shared {
				n.Layers[layerIdx].Neurons[y][x].Dimension = sharedSubNet
			} else {
				subNet := NewNetwork(layerSizes, activations, fullyConnected)
				initializeWeights(subNet, opts.InitMethod)
				n.Layers[layerIdx].Neurons[y][x].Dimension = subNet
			}
		}
	}
}

// initializeWeights applies initialization method to network weights
func initializeWeights(net *Network, method string) {
	for l := 1; l < len(net.Layers); l++ {
		for y := 0; y < net.Layers[l].Height; y++ {
			for x := 0; x < net.Layers[l].Width; x++ {
				neuron := net.Layers[l].Neurons[y][x]
				fanIn := len(neuron.Inputs)
				switch method {
				case "xavier":
					limit := math.Sqrt(6.0 / float64(fanIn+net.Layers[l].Width))
					for i := range neuron.Inputs {
						neuron.Inputs[i].Weight = rand.Float64()*2*limit - limit
					}
				case "he":
					if neuron.Activation == "relu" || neuron.Activation == "leaky_relu" {
						limit := math.Sqrt(2.0 / float64(fanIn))
						for i := range neuron.Inputs {
							neuron.Inputs[i].Weight = rand.NormFloat64() * limit
						}
					}
				default: // Random initialization as fallback
					for i := range neuron.Inputs {
						neuron.Inputs[i].Weight = rand.NormFloat64() * 0.1
					}
				}
			}
		}
	}
}
