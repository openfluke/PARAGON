// transformer.go
package paragon

import (
	"fmt"
	"math"
	"math/rand"
)

type TransformerConfig struct {
	DModel      int
	NHeads      int
	NLayers     int
	FeedForward int
	VocabSize   int
	MaxLength   int
	Activation  string
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
	layerSizes := []struct{ Width, Height int }{
		{config.MaxLength, 1},
		{config.DModel, config.MaxLength},
		{config.VocabSize, config.MaxLength},
	}
	activations := []string{"linear", config.Activation, "linear"}
	fullyConnected := []bool{true, true, true}

	n := NewNetwork(layerSizes, activations, fullyConnected)
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
				n.AttnWeights[h].QWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
				n.AttnWeights[h].KWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
				n.AttnWeights[h].VWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.DModel))
			}
		}
	}

	fmt.Printf("Layer sizes: Input=%dx%d, Hidden=%dx%d, Output=%dx%d\n",
		n.Layers[0].Width, n.Layers[0].Height,
		n.Layers[1].Width, n.Layers[1].Height,
		n.Layers[2].Width, n.Layers[2].Height)

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
		for j := range v {
			for l := 0; l < len(v[0]); l++ {
				output[i][l] += scores[i][j] * v[j][l]
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

func (n *Network) ForwardTransformer(inputs [][]float64) [][]float64 {
	n.Forward(inputs)
	hiddenLayer := n.Layers[1]
	hiddenValues := hiddenLayer.NeuronsToValues()
	pe := PositionalEncoding(hiddenLayer.Height, hiddenLayer.Width)
	for y := 0; y < len(hiddenValues); y++ {
		for x := 0; x < len(hiddenValues[y]); x++ {
			hiddenValues[y][x] += pe[y][x]
		}
		hiddenValues[y] = LayerNorm(hiddenValues[y])
	}

	headSize := hiddenLayer.Width / 4
	heads := make([][][]float64, 4)
	for h := 0; h < 4; h++ {
		qHead := make([][]float64, len(hiddenValues))
		kHead := make([][]float64, len(hiddenValues))
		vHead := make([][]float64, len(hiddenValues))
		for i := range hiddenValues {
			qHead[i] = make([]float64, headSize)
			kHead[i] = make([]float64, headSize)
			vHead[i] = make([]float64, headSize)
			for j := 0; j < headSize; j++ {
				qSum, kSum, vSum := 0.0, 0.0, 0.0
				for k := 0; k < hiddenLayer.Width; k++ {
					qSum += hiddenValues[i][k] * n.AttnWeights[h].QWeights[k][j]
					kSum += hiddenValues[i][k] * n.AttnWeights[h].KWeights[k][j]
					vSum += hiddenValues[i][k] * n.AttnWeights[h].VWeights[k][j]
				}
				qHead[i][j] = qSum
				kHead[i][j] = kSum
				vHead[i][j] = vSum
			}
		}
		heads[h], _, _, _ = ScaledDotProductAttention(qHead, kHead, vHead, headSize)
	}
	attnOutput := make([][]float64, hiddenLayer.Height)
	for y := range attnOutput {
		attnOutput[y] = make([]float64, hiddenLayer.Width)
		for h := 0; h < 4; h++ {
			start := h * headSize
			for x := 0; x < headSize; x++ {
				attnOutput[y][start+x] = heads[h][y][x]
			}
		}
		attnOutput[y] = LayerNorm(attnOutput[y])
		for x := 0; x < hiddenLayer.Width; x++ {
			attnOutput[y][x] += hiddenValues[y][x] // Residual
		}
		attnOutput[y] = LayerNorm(attnOutput[y])
	}

	// Feed-forward layer
	ffOutput := make([][]float64, hiddenLayer.Height)
	for y := range ffOutput {
		ffOutput[y] = make([]float64, hiddenLayer.Width)
		for x := 0; x < hiddenLayer.Width; x++ {
			sum := attnOutput[y][x] * 0.1 // Simple FF simulation
			ffOutput[y][x] = applyActivation(sum, "relu")
		}
		ffOutput[y] = LayerNorm(ffOutput[y])
		for x := 0; x < hiddenLayer.Width; x++ {
			ffOutput[y][x] += attnOutput[y][x] // Residual
		}
		ffOutput[y] = LayerNorm(ffOutput[y])
	}

	for y := 0; y < hiddenLayer.Height; y++ {
		for x := 0; x < hiddenLayer.Width; x++ {
			hiddenLayer.Neurons[y][x].Value = ffOutput[y][x]
		}
	}

	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			sum := neuron.Bias
			for _, conn := range neuron.Inputs {
				srcNeuron := hiddenLayer.Neurons[conn.SourceY][conn.SourceX]
				sum += srcNeuron.Value * conn.Weight
			}
			neuron.Value = applyActivation(sum, neuron.Activation)
		}
	}
	return outputLayer.NeuronsToValues()
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
