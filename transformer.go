// transformer.go
package paragon

import (
	"fmt"
	"math"
	"math/rand"
)

type TransformerConfig struct {
	DModel      int    // Dimension of the model
	NHeads      int    // Number of attention heads
	NLayers     int    // Number of layers (not used here but part of typical config)
	FeedForward int    // Size of the feed-forward intermediate layer
	VocabSize   int    // Size of the vocabulary
	MaxLength   int    // Maximum sequence length
	Activation  string // Activation function (e.g., "relu")
	GridRows    int    // Rows for 2D input (0 for 1D)
	GridCols    int    // Columns for 2D input (0 for 1D)
}

type AttentionWeights struct {
	QWeights [][]float64 // [DModel][HeadSize]
	KWeights [][]float64
	VWeights [][]float64
}

func NewTransformerEncoder(config TransformerConfig) *Network {
	if config.VocabSize <= 0 {
		panic(fmt.Sprintf("invalid VocabSize: %d", config.VocabSize))
	}
	if config.MaxLength <= 0 {
		panic(fmt.Sprintf("invalid MaxLength: %d", config.MaxLength))
	}

	// Total layers: 1 input + config.NLayers hidden + 1 output
	totalLayers := 2 + config.NLayers
	layerSizes := make([]struct{ Width, Height int }, totalLayers)
	activations := make([]string, totalLayers)
	fullyConnected := make([]bool, totalLayers)

	// Input layer: dimensions [VocabSize, MaxLength]
	layerSizes[0] = struct{ Width, Height int }{config.VocabSize, config.MaxLength}
	activations[0] = "linear"
	fullyConnected[0] = true

	// Hidden layers: use [DModel, MaxLength] for each hidden layer
	for i := 1; i <= config.NLayers; i++ {
		layerSizes[i] = struct{ Width, Height int }{config.DModel, config.MaxLength}
		activations[i] = config.Activation
		fullyConnected[i] = true
	}

	// Output layer: dimensions [VocabSize, MaxLength]
	layerSizes[totalLayers-1] = struct{ Width, Height int }{config.VocabSize, config.MaxLength}
	activations[totalLayers-1] = "linear"
	fullyConnected[totalLayers-1] = true

	// Initialize the network with the dynamic layers
	n := NewNetwork(layerSizes, activations, fullyConnected)
	n.NHeads = config.NHeads
	n.Config = config // Store the config for later use

	// Initialize attention weights for multi-head attention
	n.AttnWeights = make([]AttentionWeights, config.NHeads)
	headSize := config.DModel / config.NHeads
	for h := range n.AttnWeights {
		n.AttnWeights[h].QWeights = make([][]float64, config.DModel)
		n.AttnWeights[h].KWeights = make([][]float64, config.DModel)
		n.AttnWeights[h].VWeights = make([][]float64, config.DModel)
		for i := 0; i < config.DModel; i++ {
			n.AttnWeights[h].QWeights[i] = make([]float64, headSize)
			n.AttnWeights[h].KWeights[i] = make([]float64, headSize)
			n.AttnWeights[h].VWeights[i] = make([]float64, headSize)
			for j := 0; j < headSize; j++ {
				// Xavier/Glorot initialization scaled by DModel
				n.AttnWeights[h].QWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
				n.AttnWeights[h].KWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
				n.AttnWeights[h].VWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
			}
		}
	}

	// Initialize feed-forward weights and biases
	// FFWeights1: [DModel][FeedForward]
	n.FFWeights1 = make([][]float64, config.DModel)
	for i := range n.FFWeights1 {
		n.FFWeights1[i] = make([]float64, config.FeedForward)
		for j := range n.FFWeights1[i] {
			n.FFWeights1[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
		}
	}
	n.FFBias1 = make([]float64, config.FeedForward)

	// FFWeights2: [FeedForward][DModel]
	n.FFWeights2 = make([][]float64, config.FeedForward)
	for i := range n.FFWeights2 {
		n.FFWeights2[i] = make([]float64, config.DModel)
		for j := range n.FFWeights2[i] {
			n.FFWeights2[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.FeedForward))
		}
	}
	n.FFBias2 = make([]float64, config.DModel)

	// Initialize neuron weights and biases for all hidden and output layers
	for l := 1; l < len(n.Layers); l++ {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				neuron := n.Layers[l].Neurons[y][x]
				fanIn := len(neuron.Inputs)
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight = rand.NormFloat64() * math.Sqrt(2.0/float64(fanIn))
				}
				neuron.Bias = 0.0
			}
		}
	}

	// Debug output: print all layer sizes dynamically
	fmt.Printf("Layer sizes:")
	for i, layer := range n.Layers {
		fmt.Printf(" Layer %d: %dx%d", i, layer.Width, layer.Height)
	}
	fmt.Println()

	return n
}

func PositionalEncoding(maxLength, dModel int) [][]float64 {
	pe := make([][]float64, maxLength)
	for i := range pe {
		pe[i] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			angle := float64(i) / math.Pow(10000, float64(2*j)/float64(dModel))
			if j%2 == 0 {
				pe[i][j] = math.Sin(angle)
			} else {
				pe[i][j] = math.Cos(angle)
			}
		}
	}
	return pe
}

func ScaledDotProductAttention(q, k, v [][]float64, headSize int) ([][]float64, [][]float64, [][]float64, [][]float64) {
	d := headSize
	scores := make([][]float64, len(q))
	dScores := make([][]float64, len(q))
	for i := range q {
		scores[i] = make([]float64, len(k))
		dScores[i] = make([]float64, len(k))
		for j := range k {
			sum := 0.0
			for l := 0; l < d; l++ {
				sum += q[i][l] * k[j][l]
			}
			scores[i][j] = sum / math.Sqrt(float64(d))
		}
		scores[i], dScores[i] = SoftmaxWithGrad(scores[i])
	}
	output := make([][]float64, len(q))
	dQ := make([][]float64, len(q))
	dK := make([][]float64, len(k))
	dV := make([][]float64, len(v))
	for i := range output {
		output[i] = make([]float64, len(v[0]))
		dQ[i] = make([]float64, d)
		dK[i] = make([]float64, d)
		dV[i] = make([]float64, len(v[0]))
		for j := range v {
			for l := 0; l < len(v[0]); l++ {
				output[i][l] += scores[i][j] * v[j][l]
				dV[i][l] += scores[i][j] // Simplified gradient
			}
			for l := 0; l < d; l++ {
				dQ[i][l] += dScores[i][j] * k[j][l]
				dK[i][l] += dScores[i][j] * q[i][l]
			}
		}
	}
	return output, dQ, dK, dV
}

func SoftmaxWithGrad(inputs []float64) ([]float64, []float64) {
	maxVal := inputs[0]
	for _, v := range inputs {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expInputs := make([]float64, len(inputs))
	for i, v := range inputs {
		expInputs[i] = math.Exp(v - maxVal)
		expSum += expInputs[i]
	}
	probs := make([]float64, len(inputs))
	dProbs := make([]float64, len(inputs))
	for i := range probs {
		probs[i] = expInputs[i] / expSum
		dProbs[i] = probs[i] * (1 - probs[i]) // Simplified derivative
	}
	return probs, dProbs
}

// transformer.go (partial update)
func (n *Network) ForwardTransformer(inputs [][]float64) [][]float64 {
	// Process the input layer.
	n.Forward(inputs) // This sets values in layer 0.

	numLayers := len(n.Layers)
	if numLayers < 3 {
		panic("Network must have at least an input, one hidden, and an output layer")
	}

	// Begin with the hidden representation from the first hidden layer.
	hidden := n.Layers[1].NeuronsToValues() // shape: [MaxLength][DModel]

	// Iterate over each hidden layer block (layers 1 to numLayers-2).
	for layerIdx := 1; layerIdx < numLayers-1; layerIdx++ {
		// 1. Apply Positional Encoding and Layer Normalization.
		pe := PositionalEncoding(len(hidden), len(hidden[0]))
		for i := 0; i < len(hidden); i++ {
			for j := 0; j < len(hidden[i]); j++ {
				hidden[i][j] += pe[i][j]
			}
			hidden[i] = LayerNorm(hidden[i])
		}

		// 2. Multi-Head Attention Block.
		headSize := len(hidden[0]) / n.NHeads
		heads := make([][][]float64, n.NHeads) // Each head: [MaxLength][headSize]
		for h := 0; h < n.NHeads; h++ {
			qHead := make([][]float64, len(hidden))
			kHead := make([][]float64, len(hidden))
			vHead := make([][]float64, len(hidden))
			for i := 0; i < len(hidden); i++ {
				qHead[i] = make([]float64, headSize)
				kHead[i] = make([]float64, headSize)
				vHead[i] = make([]float64, headSize)
				for j := 0; j < headSize; j++ {
					for k := 0; k < len(hidden[i]); k++ {
						qHead[i][j] += hidden[i][k] * n.AttnWeights[h].QWeights[k][j]
						kHead[i][j] += hidden[i][k] * n.AttnWeights[h].KWeights[k][j]
						vHead[i][j] += hidden[i][k] * n.AttnWeights[h].VWeights[k][j]
					}
				}
			}
			attnOut, _, _, _ := ScaledDotProductAttention(qHead, kHead, vHead, headSize)
			heads[h] = attnOut
		}

		// Combine heads into a single attention output.
		attnOutput := make([][]float64, len(hidden))
		for i := 0; i < len(hidden); i++ {
			attnOutput[i] = make([]float64, len(hidden[0]))
			for h := 0; h < n.NHeads; h++ {
				startIdx := h * headSize
				for j := 0; j < headSize; j++ {
					attnOutput[i][startIdx+j] = heads[h][i][j]
				}
			}
			// Residual connection and layer normalization.
			for j := 0; j < len(hidden[i]); j++ {
				attnOutput[i][j] += hidden[i][j]
			}
			attnOutput[i] = LayerNorm(attnOutput[i])
		}

		// 3. Feed-Forward Block (Two-layer FFN with residual connection).
		ffOutput := make([][]float64, len(attnOutput))
		for i := 0; i < len(attnOutput); i++ {
			// First layer: from DModel to FeedForward dimension.
			intermediate := make([]float64, n.Config.FeedForward)
			for j := 0; j < n.Config.FeedForward; j++ {
				sum := n.FFBias1[j]
				for k := 0; k < len(attnOutput[i]); k++ {
					sum += n.FFWeights1[k][j] * attnOutput[i][k]
				}
				intermediate[j] = applyActivation(sum, n.Config.Activation)
			}
			// Second layer: back from FeedForward dimension to DModel.
			ffOutput[i] = make([]float64, len(attnOutput[i]))
			for j := 0; j < len(attnOutput[i]); j++ {
				sum := n.FFBias2[j]
				for k := 0; k < n.Config.FeedForward; k++ {
					sum += n.FFWeights2[k][j] * intermediate[k]
				}
				ffOutput[i][j] = sum
			}
			// Residual connection and layer normalization.
			ffOutput[i] = LayerNorm(ffOutput[i])
			for j := 0; j < len(attnOutput[i]); j++ {
				ffOutput[i][j] += attnOutput[i][j]
			}
			ffOutput[i] = LayerNorm(ffOutput[i])
		}

		// Set hidden to the output of this block.
		hidden = ffOutput
	}

	// Use the final hidden representation (from layer index numLayers-2) for the output layer.
	outputLayer := n.Layers[numLayers-1]
	output := make([][]float64, 1)
	output[0] = make([]float64, outputLayer.Height*outputLayer.Width)
	idx := 0
	lastHidden := hidden // final hidden representation
	for i := 0; i < outputLayer.Height; i++ {
		for j := 0; j < outputLayer.Width; j++ {
			neuron := outputLayer.Neurons[i][j]
			sum := neuron.Bias
			for _, conn := range neuron.Inputs {
				// Here we assume the connection's SourceY and SourceX index into lastHidden.
				srcVal := lastHidden[conn.SourceY][conn.SourceX]
				sum += srcVal * conn.Weight
			}
			neuron.Value = applyActivation(sum, neuron.Activation)
			output[0][idx] = neuron.Value
			idx++
		}
	}
	return output
}

func (n *Grid) NeuronsToValues() [][]float64 {
	values := make([][]float64, n.Height)
	for y := 0; y < n.Height; y++ {
		values[y] = make([]float64, n.Width)
		for x := 0; x < n.Width; x++ {
			values[y][x] = n.Neurons[y][x].Value
		}
	}
	return values
}

func LayerNorm(values []float64) []float64 {
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	std := math.Sqrt(variance + 1e-5)
	norm := make([]float64, len(values))
	for i, v := range values {
		norm[i] = (v - mean) / std
	}
	return norm
}

func PositionalEncoding2D(rows, cols, dModel int) [][]float64 {
	pe := make([][]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		r := i / cols
		c := i % cols
		pe[i] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			angleR := float64(r) / math.Pow(10000, float64(2*j)/float64(dModel))
			angleC := float64(c) / math.Pow(10000, float64(2*j)/float64(dModel))
			if j%2 == 0 {
				pe[i][j] = math.Sin(angleR) + math.Sin(angleC)
			} else {
				pe[i][j] = math.Cos(angleR) + math.Cos(angleC)
			}
		}
	}
	return pe
}
