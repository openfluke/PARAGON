package paragon

import (
	"fmt"
	"math/rand"
)

// PerturbWeights adds random noise to all connection weights in the network.
// Works for all numeric types. Uses Gaussian noise N(0, rate).
func (n *Network[T]) PerturbWeights(rate float64, seed int) {
	rng := rand.New(rand.NewSource(int64(seed)))

	for l := 1; l < len(n.Layers); l++ {
		layer := n.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				for k := range neuron.Inputs {
					weight := neuron.Inputs[k].Weight

					switch any(weight).(type) {

					case float32, float64:
						noise := rng.NormFloat64() * rate
						neuron.Inputs[k].Weight += T(noise)

					case int:
						v := int(weight)
						noise := int(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight = T(v + noise)

					case int8:
						v := int8(weight)
						noise := int8(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight = T(v + noise)

					case int16:
						v := int16(weight)
						noise := int16(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight = T(v + noise)

					case int32:
						v := int32(weight)
						noise := int32(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight = T(v + noise)

					case int64:
						v := int64(weight)
						noise := int64(rng.NormFloat64() * rate * 10)
						neuron.Inputs[k].Weight = T(v + noise)

					case uint:
						v := uint(weight)
						noise := int(rng.NormFloat64() * rate * 10)
						newVal := int(v) + noise
						if newVal < 0 {
							newVal = 0
						}
						neuron.Inputs[k].Weight = T(uint(newVal))

					case uint8:
						v := uint8(weight)
						noise := int(rng.NormFloat64() * rate * 10)
						newVal := int(v) + noise
						if newVal < 0 {
							newVal = 0
						}
						if newVal > 255 {
							newVal = 255
						}
						neuron.Inputs[k].Weight = T(uint8(newVal))

					case uint16:
						v := uint16(weight)
						noise := int(rng.NormFloat64() * rate * 10)
						newVal := int(v) + noise
						if newVal < 0 {
							newVal = 0
						}
						if newVal > 65535 {
							newVal = 65535
						}
						neuron.Inputs[k].Weight = T(uint16(newVal))

					case uint32:
						v := uint32(weight)
						noise := int(rng.NormFloat64() * rate * 10)
						newVal := int64(v) + int64(noise)
						if newVal < 0 {
							newVal = 0
						}
						if newVal > int64(^uint32(0)) {
							newVal = int64(^uint32(0))
						}
						neuron.Inputs[k].Weight = T(uint32(newVal))

					case uint64:
						v := uint64(weight)
						noise := int64(rng.NormFloat64() * rate * 10)
						newVal := int64(v) + noise
						if newVal < 0 {
							newVal = 0
						}
						neuron.Inputs[k].Weight = T(uint64(newVal))

					default:
						fmt.Printf("⚠️ Unknown type in PerturbWeights: %T\n", weight)
					}
				}
			}
		}
	}
}
