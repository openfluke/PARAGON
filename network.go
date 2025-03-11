package paragon

import (
	"fmt"
	"math"
	"math/rand"
)

// Grid represents a 2D layer of neurons
type Grid struct {
	Width   int         // Number of neurons along x-axis
	Height  int         // Number of neurons along y-axis
	Neurons [][]*Neuron // 2D slice of neuron pointers
}

// Neuron represents a single unit in the grid
type Neuron struct {
	ID         int          // Unique ID
	Value      float64      // Current activation
	Bias       float64      // Bias term
	Activation string       // e.g., "relu", "sigmoid"
	Type       string       // e.g., "dense", extensible for future types
	Inputs     []Connection // Incoming connections
	IsNew      bool         // Flag for newly added neurons
}

// Connection defines a link from a source neuron
type Connection struct {
	SourceLayer int     // Layer index of source neuron
	SourceX     int     // X-coordinate of source neuron
	SourceY     int     // Y-coordinate of source neuron
	Weight      float64 // Weight of the connection
}

// Network encapsulates the entire model
type Network struct {
	Layers      []Grid
	InputLayer  int
	OutputLayer int
	Debug       bool
	AttnWeights []AttentionWeights // Per head
	NHeads      int
	FFWeights1  [][]float64       // [DModel][FeedForward]
	FFBias1     []float64         // [FeedForward]
	FFWeights2  [][]float64       // [FeedForward][DModel]
	FFBias2     []float64         // [DModel]
	Config      TransformerConfig // Configuration settings
}

// NewNetwork initializes a network with specified layer sizes, activations, and connectivity
func NewNetwork(layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool) *Network {
	if len(layerSizes) != len(activations) || len(layerSizes) != len(fullyConnected) {
		panic("mismatched layer sizes, activations, or connectivity settings")
	}
	n := &Network{
		Layers:      make([]Grid, len(layerSizes)),
		InputLayer:  0,
		OutputLayer: len(layerSizes) - 1,
		NHeads:      0, // Default to 0, set by NewTransformerEncoder if needed
	}
	idCounter := 0
	for i, size := range layerSizes {
		grid := Grid{
			Width:   size.Width,
			Height:  size.Height,
			Neurons: make([][]*Neuron, size.Height),
		}
		for y := 0; y < size.Height; y++ {
			grid.Neurons[y] = make([]*Neuron, size.Width)
			for x := 0; x < size.Width; x++ {
				grid.Neurons[y][x] = &Neuron{
					ID:         idCounter,
					Bias:       0.0,
					Activation: activations[i],
					Type:       "dense",
				}
				idCounter++
			}
		}
		n.Layers[i] = grid
	}
	n.ConnectLayers(fullyConnected)
	return n
}

func (n *Network) getFullyConnectedInputs(srcLayer int, srcGrid Grid) []Connection {
	fanIn := srcGrid.Width * srcGrid.Height
	conns := make([]Connection, 0, fanIn)
	for y := 0; y < srcGrid.Height; y++ {
		for x := 0; x < srcGrid.Width; x++ {
			conns = append(conns, Connection{
				SourceLayer: srcLayer,
				SourceX:     x,
				SourceY:     y,
				Weight:      rand.NormFloat64() * math.Sqrt(2.0/float64(fanIn)),
			})
		}
	}
	return conns
}

func (n *Network) getLocalConnections(srcLayer, centerX, centerY, size, stride int) []Connection {
	half := size / 2
	srcGrid := n.Layers[srcLayer]
	conns := []Connection{}
	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			srcX := centerX*stride + dx
			srcY := centerY*stride + dy
			if srcX >= 0 && srcX < srcGrid.Width && srcY >= 0 && srcY < srcGrid.Height {
				conns = append(conns, Connection{
					SourceLayer: srcLayer,
					SourceX:     srcX,
					SourceY:     srcY,
					Weight:      rand.NormFloat64() * math.Sqrt(2.0/float64(size*size)),
				})
			}
		}
	}
	return conns
}

// ConnectLayers sets up connections based on connectivity settings
func (n *Network) ConnectLayers(fullyConnected []bool) {
	for l := 1; l < len(n.Layers); l++ {
		prevLayer := n.Layers[l-1]
		currLayer := n.Layers[l]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				if fullyConnected[l] {
					neuron.Inputs = n.getFullyConnectedInputs(l-1, prevLayer)
				} else {
					// Use 5x5 receptive field with stride 1
					neuron.Inputs = n.getLocalConnections(l-1, x, y, 5, 1)
				}
			}
		}
	}
}

// Forward propagates inputs through the network
func (n *Network) Forward(inputs [][]float64) {
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
				if n.Debug {
					fmt.Printf("Layer %d, Neuron (%d,%d): Value=%.4f\n", l, x, y, neuron.Value)
				}
			}
		}
	}
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// Backward performs backpropagation
// network.go (partial update)
func (n *Network) Backward(targets [][]float64, learningRate float64) {
	errorTerms := make([][][]float64, len(n.Layers))
	for l := range n.Layers {
		errorTerms[l] = make([][]float64, n.Layers[l].Height)
		for y := range errorTerms[l] {
			errorTerms[l][y] = make([]float64, n.Layers[l].Width)
		}
	}

	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			errorTerms[n.OutputLayer][y][x] = (targets[y][x] - neuron.Value) * activationDerivative(neuron.Value, neuron.Activation)
		}
	}

	// Only compute attention gradients if this is a transformer network
	var attnGradients [][][]float64
	if n.NHeads > 0 && len(n.AttnWeights) > 0 {
		hiddenLayer := n.Layers[1]
		headSize := hiddenLayer.Width / n.NHeads
		attnGradients = make([][][]float64, n.NHeads)
		for h := 0; h < n.NHeads; h++ {
			attnGradients[h] = make([][]float64, hiddenLayer.Height)
			for y := 0; y < hiddenLayer.Height; y++ {
				attnGradients[h][y] = make([]float64, headSize)
				start := h * headSize
				for x := 0; x < headSize; x++ {
					attnGradients[h][y][x] = errorTerms[1][y][start+x]
				}
			}
		}
	}

	for l := n.OutputLayer; l > 0; l-- {
		currLayer := n.Layers[l]
		prevLayer := n.Layers[l-1]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				errorTerm := errorTerms[l][y][x]
				neuron.Bias += learningRate * errorTerm
				for i, conn := range neuron.Inputs {
					sourceNeuron := prevLayer.Neurons[conn.SourceY][conn.SourceX]
					gradient := errorTerm * sourceNeuron.Value
					if gradient > 5.0 {
						gradient = 5.0
					} else if gradient < -5.0 {
						gradient = -5.0
					}
					neuron.Inputs[i].Weight += learningRate * gradient
					if l-1 > 0 {
						errorTerms[l-1][conn.SourceY][conn.SourceX] += errorTerm * conn.Weight
					}
				}
			}
		}
		if l-1 > 0 {
			for y := 0; y < prevLayer.Height; y++ {
				for x := 0; x < prevLayer.Width; x++ {
					errorTerms[l-1][y][x] *= activationDerivative(prevLayer.Neurons[y][x].Value, prevLayer.Neurons[y][x].Activation)
				}
			}
			// Transformer-specific attention weight updates
			if n.NHeads > 0 && len(n.AttnWeights) > 0 {
				hiddenLayer := n.Layers[1]
				headSize := hiddenLayer.Width / n.NHeads
				for h := 0; h < n.NHeads; h++ {
					for i := 0; i < hiddenLayer.Width; i++ {
						for j := 0; j < headSize; j++ {
							gradientQ, gradientK, gradientV := 0.0, 0.0, 0.0
							for y := 0; y < hiddenLayer.Height; y++ {
								gradientQ += attnGradients[h][y][j] * hiddenLayer.Neurons[y][i].Value
								gradientK += attnGradients[h][y][j] * hiddenLayer.Neurons[y][i].Value
								gradientV += attnGradients[h][y][j] * hiddenLayer.Neurons[y][i].Value
							}
							if gradientQ > 5.0 {
								gradientQ = 5.0
							} else if gradientQ < -5.0 {
								gradientQ = -5.0
							}
							if gradientK > 5.0 {
								gradientK = 5.0
							} else if gradientK < -5.0 {
								gradientK = -5.0
							}
							if gradientV > 5.0 {
								gradientV = 5.0
							} else if gradientV < -5.0 {
								gradientV = -5.0
							}
							n.AttnWeights[h].QWeights[i][j] += learningRate * gradientQ
							n.AttnWeights[h].KWeights[i][j] += learningRate * gradientK
							n.AttnWeights[h].VWeights[i][j] += learningRate * gradientV
						}
					}
				}
			}

		}
	}
}

// Train runs the training loop
func (n *Network) Train(inputs [][][]float64, targets [][][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))
		shuffledInputs := make([][][]float64, len(inputs))
		shuffledTargets := make([][][]float64, len(targets))
		for i, p := range perm {
			shuffledInputs[i] = inputs[p]
			shuffledTargets[i] = targets[p]
		}
		for b := 0; b < len(shuffledInputs); b++ {
			n.Forward(shuffledInputs[b])
			loss := n.ComputeLoss(shuffledTargets[b])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			totalLoss += loss
			n.Backward(shuffledTargets[b], learningRate)
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

// ComputeLoss calculates the loss for a sample
func (n *Network) ComputeLoss(target [][]float64) float64 {
	loss := 0.0
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			outputVal := outputLayer.Neurons[y][x].Value
			targetVal := target[y][x]
			if outputVal <= 0 {
				outputVal = 1e-10
			}
			loss += -targetVal * math.Log(outputVal)
		}
	}
	return loss
}

// AddNeuronsToLayer adds neurons to a specified layer
func (n *Network) AddNeuronsToLayer(layerIdx, numToAdd int) {
	if layerIdx <= n.InputLayer || layerIdx > n.OutputLayer {
		return
	}
	grid := &n.Layers[layerIdx]
	newHeight := grid.Height + numToAdd
	newNeurons := make([][]*Neuron, newHeight)
	copy(newNeurons, grid.Neurons)
	for y := grid.Height; y < newHeight; y++ {
		newNeurons[y] = make([]*Neuron, grid.Width)
		for x := 0; x < grid.Width; x++ {
			newNeurons[y][x] = &Neuron{
				ID:         n.getNextID(),
				Bias:       0.0,
				Activation: grid.Neurons[0][0].Activation,
				Type:       "dense",
				IsNew:      true,
			}
			prevLayer := n.Layers[layerIdx-1]
			if layerIdx == 1 && len(n.Layers) == 3 { // Initial setup check
				newNeurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
			} else {
				newNeurons[y][x].Inputs = n.getLocalConnections(layerIdx-1, x, y, 3, 1) // Added stride 1
			}
		}
	}
	grid.Neurons = newNeurons
	grid.Height = newHeight

	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := len(nextLayer.Neurons[0][0].Inputs) == grid.Width*grid.Height
		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, *grid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 3, 1) // Added stride 1
				}
			}
		}
	}
	if n.Debug {
		fmt.Printf("Added %d neurons to layer %d, new height: %d\n", numToAdd, layerIdx, grid.Height)
	}
}

// getNextID generates a unique neuron ID
func (n *Network) getNextID() int {
	maxID := 0
	for _, layer := range n.Layers {
		for _, row := range layer.Neurons {
			for _, neuron := range row {
				if neuron.ID > maxID {
					maxID = neuron.ID
				}
			}
		}
	}
	return maxID + 1
}

// ApplySoftmax normalizes output layer values
func (n *Network) ApplySoftmax() {
	outputGrid := n.Layers[n.OutputLayer]
	values := make([]float64, outputGrid.Width*outputGrid.Height)
	idx := 0
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			values[idx] = outputGrid.Neurons[y][x].Value
			idx++
		}
	}
	softmaxValues := Softmax(values)
	idx = 0
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			outputGrid.Neurons[y][x].Value = softmaxValues[idx]
			idx++
		}
	}
}

// AddLayer inserts a new layer at the specified index with given dimensions and connectivity
func (n *Network) AddLayer(layerIdx int, width, height int, activation string, fullyConnectedToPrev bool) {
	if layerIdx < n.InputLayer || layerIdx > n.OutputLayer+1 {
		return
	}
	newGrid := Grid{
		Width:   width,
		Height:  height,
		Neurons: make([][]*Neuron, height),
	}
	idCounter := n.getNextID()
	for y := 0; y < height; y++ {
		newGrid.Neurons[y] = make([]*Neuron, width)
		for x := 0; x < width; x++ {
			newGrid.Neurons[y][x] = &Neuron{
				ID:         idCounter,
				Bias:       0.0,
				Activation: activation,
				Type:       "dense",
			}
			idCounter++
		}
	}
	newLayers := make([]Grid, len(n.Layers)+1)
	copy(newLayers[:layerIdx], n.Layers[:layerIdx])
	newLayers[layerIdx] = newGrid
	copy(newLayers[layerIdx+1:], n.Layers[layerIdx:])
	n.Layers = newLayers
	n.OutputLayer = len(n.Layers) - 1

	if layerIdx > n.InputLayer {
		prevLayer := n.Layers[layerIdx-1]
		for y := 0; y < newGrid.Height; y++ {
			for x := 0; x < newGrid.Width; x++ {
				if fullyConnectedToPrev {
					newGrid.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
				} else {
					newGrid.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx-1, x, y, 5, 1)
				}
			}
		}
	}
	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := layerIdx == n.OutputLayer-1
		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, newGrid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 5, 1)
				}
			}
		}
	}
	if n.Debug {
		fmt.Printf("Added new layer at index %d with dimensions %dx%d\n", layerIdx, width, height)
	}
}
