package paragon

import (
	"fmt"
	"math/rand"
)

// PerturbWeights adds random noise to all connection weights in the network.
// Works for all numeric types. Uses Gaussian noise N(0, rate).
func (n *Network[T]) PerturbWeights(rate float64, seed int) {
	rng := rand.New(rand.NewSource(int64(seed)))

	for l := 1; l < len(n.Layers); l++ { // Skip input layer
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				for k := range neuron.Inputs {
					curr := neuron.Inputs[k].Weight

					switch any(curr).(type) {
					case float32, float64:
						noise := rng.NormFloat64() * rate
						neuron.Inputs[k].Weight += T(noise)

					case int, int8, int16, int32, int64:
						noise := int(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight += T(noise)

					case uint, uint8, uint16, uint32, uint64:
						noise := int(rng.NormFloat64() * rate * 10)
						newVal := int(any(curr).(uint)) + noise
						if newVal < 0 {
							newVal = 0
						}
						neuron.Inputs[k].Weight = T(newVal)

					default:
						fmt.Printf("⚠️ Unknown type in PerturbWeights: %T\n", curr)
					}
				}
			}
		}
	}
}
