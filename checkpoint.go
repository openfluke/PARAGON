package paragon

import (
	"encoding/json"
	"fmt"
	"os"
)

// GetLayerState returns the current values of all neurons in the specified layer.
func (n *Network) GetLayerState(layerIdx int) [][]float64 {
	if layerIdx < 0 || layerIdx >= len(n.Layers) {
		panic(fmt.Sprintf("invalid layer index: %d", layerIdx))
	}
	layer := n.Layers[layerIdx]
	state := make([][]float64, layer.Height)
	for y := 0; y < layer.Height; y++ {
		state[y] = make([]float64, layer.Width)
		for x := 0; x < layer.Width; x++ {
			state[y][x] = layer.Neurons[y][x].Value
		}
	}
	return state
}

// ForwardFromLayer computes the forward pass starting from the specified layer using the provided state.
func (n *Network) ForwardFromLayer(layerIdx int, layerState [][]float64) {
	// Validate inputs
	if layerIdx < 0 || layerIdx >= n.OutputLayer {
		panic(fmt.Sprintf("invalid layer index for checkpoint: %d", layerIdx))
	}
	if len(layerState) != n.Layers[layerIdx].Height || len(layerState[0]) != n.Layers[layerIdx].Width {
		panic(fmt.Sprintf("mismatched dimensions for layer state: expected %dx%d, got %dx%d",
			n.Layers[layerIdx].Height, n.Layers[layerIdx].Width, len(layerState), len(layerState[0])))
	}
	// Set layerIdx values
	for y := 0; y < n.Layers[layerIdx].Height; y++ {
		for x := 0; x < n.Layers[layerIdx].Width; x++ {
			n.Layers[layerIdx].Neurons[y][x].Value = layerState[y][x]
		}
	}
	// Compute subsequent layers
	for l := layerIdx + 1; l <= n.OutputLayer; l++ {
		currLayer := n.Layers[l]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				sum := neuron.Bias
				for _, conn := range neuron.Inputs {
					srcNeuron := n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
					sum += srcNeuron.Value * conn.Weight
				}
				neuron.Value = applyActivation(sum, neuron.Activation)
			}
		}
	}
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// SaveLayerState saves the state of the specified layer to a JSON file.
func (n *Network) SaveLayerState(layerIdx int, filename string) error {
	// Capture the current state of the layer
	state := n.GetLayerState(layerIdx)

	// Serialize the state to JSON
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("failed to marshal layer state: %v", err)
	}

	// Write the JSON data to the specified file
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write layer state to file: %v", err)
	}
	return nil
}

// LoadLayerState loads the state of the specified layer from a JSON file.
func (n *Network) LoadLayerState(layerIdx int, filename string) ([][]float64, error) {
	// Read the JSON file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read layer state from file: %v", err)
	}

	// Deserialize the JSON data into a 2D slice
	var state [][]float64
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal layer state: %v", err)
	}

	// Validate that the dimensions match the layer's expected size
	if len(state) != n.Layers[layerIdx].Height || len(state[0]) != n.Layers[layerIdx].Width {
		return nil, fmt.Errorf("mismatched dimensions: expected %dx%d, got %dx%d",
			n.Layers[layerIdx].Height, n.Layers[layerIdx].Width, len(state), len(state[0]))
	}
	return state, nil
}

// ForwardUntilLayer runs the forward pass only up to the given layer index (inclusive)
// and returns the state of that layer.
func (n *Network) ForwardUntilLayer(input [][]float64, stopLayer int) {
	// Set the input values
	inputGrid := n.Layers[n.InputLayer]
	if len(input) != inputGrid.Height || len(input[0]) != inputGrid.Width {
		panic(fmt.Sprintf("input dimensions mismatch: expected %dx%d, got %dx%d",
			inputGrid.Height, inputGrid.Width, len(input), len(input[0])))
	}
	for y := 0; y < inputGrid.Height; y++ {
		for x := 0; x < inputGrid.Width; x++ {
			inputGrid.Neurons[y][x].Value = input[y][x]
		}
	}

	// Propagate forward layer by layer until stopLayer (inclusive)
	for l := 1; l <= stopLayer; l++ {
		currLayer := n.Layers[l]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				sum := neuron.Bias
				// Sum over all inputs from previous layer
				for _, conn := range neuron.Inputs {
					srcNeuron := n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
					sum += srcNeuron.Value * conn.Weight
				}
				neuron.Value = applyActivation(sum, neuron.Activation)
			}
		}
		// If the current layer uses softmax, apply it.
		if l == n.OutputLayer && n.Layers[l].Neurons[0][0].Activation == "softmax" {
			n.ApplySoftmax()
		}
	}
}
