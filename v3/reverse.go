package paragon

import (
	"fmt"
	"math"
	"math/rand"
)

// ReverseExact attempts to reconstruct the input from the output for a linear network.
// It uses transposed weight propagation for fully connected layers with equal sizes.
func (n *Network[T]) ReverseExact(outputValues [][]float64) ([][]float64, error) {
	// Validate output dimensions
	outLayer := n.Layers[n.OutputLayer]
	if len(outputValues) != outLayer.Height || len(outputValues[0]) != outLayer.Width {
		return nil, fmt.Errorf("output mismatch: want %dx%d, got %dx%d",
			outLayer.Height, outLayer.Width, len(outputValues), len(outputValues[0]))
	}

	// Initialize reverse values to zero
	for l := range n.Layers {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				n.Layers[l].Neurons[y][x].RevValue = T(0)
			}
		}
	}

	// Set output layer reverse values
	for y := 0; y < outLayer.Height; y++ {
		for x := 0; x < outLayer.Width; x++ {
			outLayer.Neurons[y][x].RevValue = T(outputValues[y][x])
		}
	}

	// Propagate backward layer by layer
	for l := n.OutputLayer; l > n.InputLayer; l-- {
		currLayer := n.Layers[l]
		prevLayer := n.Layers[l-1]

		// Verify linear activation
		act := currLayer.Neurons[0][0].Activation
		if act != "linear" && act != "identity" {
			return nil, fmt.Errorf("ReverseExact requires linear activations, got %s at layer %d", act, l)
		}

		// Subtract bias
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				neuron.RevValue -= neuron.Bias
			}
		}

		// Check for square layer sizes
		if currLayer.Width != prevLayer.Width || currLayer.Height != prevLayer.Height {
			return nil, fmt.Errorf("ReverseExact requires equal layer sizes, got %dx%d -> %dx%d at layer %d",
				prevLayer.Width, prevLayer.Height, currLayer.Width, currLayer.Height, l)
		}

		// Propagate to previous layer using transposed weights
		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				var sum T
				for cy := 0; cy < currLayer.Height; cy++ {
					for cx := 0; cx < currLayer.Width; cx++ {
						neuron := currLayer.Neurons[cy][cx]
						for _, conn := range neuron.Inputs {
							if conn.SourceLayer == l-1 && conn.SourceX == x && conn.SourceY == y {
								sum += neuron.RevValue * conn.Weight // Use weight as in transpose
							}
						}
					}
				}
				prevLayer.Neurons[y][x].RevValue = sum
			}
		}
	}

	// Extract reconstructed input
	inLayer := n.Layers[n.InputLayer]
	reconstructed := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		reconstructed[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			reconstructed[y][x] = float64(inLayer.Neurons[y][x].RevValue)
		}
	}

	return reconstructed, nil
}

// ReversePropagate attempts to reconstruct input values by propagating output values
func (n *Network[T]) ReversePropagate(outputValues [][]float64) [][]float64 {
	// Initialize all neurons' reverse values to zero
	for l := range n.Layers {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				n.Layers[l].Neurons[y][x].RevValue = T(0)
			}
		}
	}

	// Set output layer values
	outLayer := n.Layers[n.OutputLayer]
	if len(outputValues) != outLayer.Height || len(outputValues[0]) != outLayer.Width {
		panic(fmt.Sprintf("output mismatch: want %dx%d, got %dx%d",
			outLayer.Height, outLayer.Width, len(outputValues), len(outputValues[0])))
	}

	for y := 0; y < outLayer.Height; y++ {
		for x := 0; x < outLayer.Width; x++ {
			outLayer.Neurons[y][x].RevValue = T(outputValues[y][x])
		}
	}

	// Propagate backward layer by layer
	for l := n.OutputLayer; l > n.InputLayer; l-- {
		currLayer := &n.Layers[l]

		// For each neuron in current layer, propagate its value back
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]

				// Apply inverse activation
				revVal := n.inverseActivation(neuron.RevValue, neuron.Activation)

				// Subtract bias
				revVal -= neuron.Bias

				// Propagate to input neurons weighted by connection strength
				totalWeight := T(0)
				for _, conn := range neuron.Inputs {
					totalWeight += conn.Weight * conn.Weight // Use squared weights for normalization
				}

				if totalWeight > T(0) {
					for _, conn := range neuron.Inputs {
						if conn.SourceLayer >= 0 && conn.SourceLayer < len(n.Layers) {
							// Weighted contribution
							contribution := revVal * conn.Weight / totalWeight
							n.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX].RevValue += contribution
						}
					}
				}
			}
		}
	}

	// Extract reconstructed input
	inLayer := n.Layers[n.InputLayer]
	reconstructed := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		reconstructed[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			reconstructed[y][x] = float64(inLayer.Neurons[y][x].RevValue)
		}
	}

	return reconstructed
}

// ReverseUsingGradient uses gradient-based optimization to find inputs that produce the target output
func (n *Network[T]) ReverseUsingGradient(targetOutput [][]float64, iterations int) [][]float64 {
	initScale, lr, _, err := getInitScaleAndLr[T]()
	if err != nil {
		// Fallback to default if T is not float
		return n.ReversePropagate(targetOutput)
	}

	inLayer := n.Layers[n.InputLayer]
	currentInput := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		currentInput[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			currentInput[y][x] = float64(T(rand.Float64()-0.5) * initScale)
		}
	}

	for iter := 0; iter < iterations; iter++ {
		n.Forward(currentInput)
		gradients := make([][]float64, inLayer.Height)
		for y := 0; y < inLayer.Height; y++ {
			gradients[y] = make([]float64, inLayer.Width)
		}
		epsilon := 0.0001
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				original := currentInput[y][x]
				currentInput[y][x] = original + epsilon
				n.Forward(currentInput)
				output1 := n.GetOutput()
				loss1 := 0.0
				for i := range output1 {
					diff := output1[i] - targetOutput[0][i]
					loss1 += diff * diff
				}
				currentInput[y][x] = original - epsilon
				n.Forward(currentInput)
				output2 := n.GetOutput()
				loss2 := 0.0
				for i := range output2 {
					diff := output2[i] - targetOutput[0][i]
					loss2 += diff * diff
				}
				gradients[y][x] = (loss1 - loss2) / (2 * epsilon)
				currentInput[y][x] = original
			}
		}
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				currentInput[y][x] -= float64(lr) * gradients[y][x]
			}
		}
		n.Forward(currentInput)
		currentOutput := n.GetOutput()
		currentError := 0.0
		for i := range currentOutput {
			diff := currentOutput[i] - targetOutput[0][i]
			currentError += diff * diff
		}
		if iter%100 == 0 && n.Debug {
			fmt.Printf("Iteration %d, Error: %.6f\n", iter, currentError)
		}
	}
	return currentInput
}

// inverseActivation attempts to apply the inverse of an activation function
func (n *Network[T]) inverseActivation(value T, activation string) T {
	v := float64(value)
	switch activation {
	case "linear", "identity":
		return value
	case "sigmoid":
		if v <= 0 || v >= 1 {
			if v <= 0 {
				v = 0.001
			} else {
				v = 0.999
			}
		}
		return T(math.Log(v / (1 - v)))
	case "tanh":
		if v <= -1 || v >= 1 {
			if v <= -1 {
				v = -0.999
			} else {
				v = 0.999
			}
		}
		return T(0.5 * math.Log((1+v)/(1-v)))
	case "relu":
		if v < 0 {
			return T(0)
		}
		return value
	case "leaky_relu":
		alpha := 0.01
		if v < 0 {
			return T(v / alpha)
		}
		return value
	case "softmax":
		if v <= 0 {
			v = 0.001
		}
		return T(math.Log(v))
	default:
		return value
	}
}

// ReverseUsingAdaptiveGradient uses gradient-based optimization with adaptive learning rate
func (n *Network[T]) ReverseUsingAdaptiveGradient(targetOutput [][]float64, iterations int) [][]float64 {
	initScale, initialLR, _, err := getInitScaleAndLr[T]()
	if err != nil {
		return n.ReversePropagate(targetOutput)
	}

	inLayer := n.Layers[n.InputLayer]
	currentInput := make([][]float64, inLayer.Height)
	bestInput := make([][]float64, inLayer.Height)
	momentum := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		currentInput[y] = make([]float64, inLayer.Width)
		bestInput[y] = make([]float64, inLayer.Width)
		momentum[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			currentInput[y][x] = float64(T(rand.Float64()-0.5) * initScale)
			bestInput[y][x] = currentInput[y][x]
		}
	}

	lr := float64(initialLR)
	bestError := math.Inf(1)
	momentumFactor := 0.9
	errorHistory := make([]float64, 0, 10)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([][]float64, inLayer.Height)
		for y := 0; y < inLayer.Height; y++ {
			gradients[y] = make([]float64, inLayer.Width)
		}
		epsilon := 0.0001
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				original := currentInput[y][x]
				currentInput[y][x] = original + epsilon
				n.Forward(currentInput)
				output1 := n.GetOutput()
				loss1 := 0.0
				for i := range output1 {
					diff := output1[i] - targetOutput[0][i]
					loss1 += diff * diff
				}
				currentInput[y][x] = original - epsilon
				n.Forward(currentInput)
				output2 := n.GetOutput()
				loss2 := 0.0
				for i := range output2 {
					diff := output2[i] - targetOutput[0][i]
					loss2 += diff * diff
				}
				gradients[y][x] = (loss1 - loss2) / (2 * epsilon)
				currentInput[y][x] = original
			}
		}
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				momentum[y][x] = momentumFactor*momentum[y][x] - lr*gradients[y][x]
				currentInput[y][x] += momentum[y][x]
			}
		}
		n.Forward(currentInput)
		currentOutput := n.GetOutput()
		currentError := 0.0
		for i := range currentOutput {
			diff := currentOutput[i] - targetOutput[0][i]
			currentError += diff * diff
		}
		currentError = math.Sqrt(currentError)
		if currentError < bestError {
			bestError = currentError
			for y := 0; y < inLayer.Height; y++ {
				copy(bestInput[y], currentInput[y])
			}
		}
		errorHistory = append(errorHistory, currentError)
		if len(errorHistory) > 10 {
			errorHistory = errorHistory[1:]
		}
		if len(errorHistory) >= 5 {
			recentImprovement := (errorHistory[0] - errorHistory[len(errorHistory)-1]) / errorHistory[0]
			if recentImprovement < 0.01 {
				lr *= 0.95
			} else if recentImprovement > 0.1 {
				lr *= 1.05
			}
			if lr > float64(initialLR)*2 {
				lr = float64(initialLR) * 2
			} else if lr < float64(initialLR)*0.01 {
				lr = float64(initialLR) * 0.01
			}
		}
		if currentError < 0.0001 {
			if n.Debug {
				fmt.Printf("Early stopping at iteration %d with error %.6f\n", iter, currentError)
			}
			break
		}
		if iter%100 == 0 && n.Debug {
			fmt.Printf("Iteration %d, Error: %.6f, LR: %.6f\n", iter, currentError, lr)
		}
	}
	return bestInput
}

// ReverseUsingAdam uses the Adam optimizer for better convergence
func (n *Network[T]) ReverseUsingAdam(targetOutput [][]float64, iterations int) [][]float64 {
	initScale, lr, _, err := getInitScaleAndLr[T]()
	if err != nil {
		return n.ReversePropagate(targetOutput)
	}

	inLayer := n.Layers[n.InputLayer]
	currentInput := make([][]float64, inLayer.Height)
	bestInput := make([][]float64, inLayer.Height)
	m := make([][]float64, inLayer.Height)
	v := make([][]float64, inLayer.Height)
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-8

	for y := 0; y < inLayer.Height; y++ {
		currentInput[y] = make([]float64, inLayer.Width)
		bestInput[y] = make([]float64, inLayer.Width)
		m[y] = make([]float64, inLayer.Width)
		v[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			currentInput[y][x] = float64(T(rand.Float64()-0.5) * initScale)
			bestInput[y][x] = currentInput[y][x]
		}
	}

	bestError := math.Inf(1)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([][]float64, inLayer.Height)
		for y := 0; y < inLayer.Height; y++ {
			gradients[y] = make([]float64, inLayer.Width)
		}
		gradEpsilon := 0.00001
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				original := currentInput[y][x]
				currentInput[y][x] = original + gradEpsilon
				n.Forward(currentInput)
				output1 := n.GetOutput()
				loss1 := 0.0
				for i := range output1 {
					diff := output1[i] - targetOutput[0][i]
					loss1 += diff * diff
				}
				currentInput[y][x] = original - gradEpsilon
				n.Forward(currentInput)
				output2 := n.GetOutput()
				loss2 := 0.0
				for i := range output2 {
					diff := output2[i] - targetOutput[0][i]
					loss2 += diff * diff
				}
				gradients[y][x] = (loss1 - loss2) / (2 * gradEpsilon)
				currentInput[y][x] = original
			}
		}
		t := float64(iter + 1)
		for y := 0; y < inLayer.Height; y++ {
			for x := 0; x < inLayer.Width; x++ {
				g := gradients[y][x]
				m[y][x] = beta1*m[y][x] + (1-beta1)*g
				v[y][x] = beta2*v[y][x] + (1-beta2)*g*g
				mHat := m[y][x] / (1 - math.Pow(beta1, t))
				vHat := v[y][x] / (1 - math.Pow(beta2, t))
				currentInput[y][x] -= float64(lr) * mHat / (math.Sqrt(vHat) + epsilon)
				if currentInput[y][x] > 5.0 {
					currentInput[y][x] = 5.0
				} else if currentInput[y][x] < -5.0 {
					currentInput[y][x] = -5.0
				}
			}
		}
		n.Forward(currentInput)
		currentOutput := n.GetOutput()
		currentError := 0.0
		for i := range currentOutput {
			diff := currentOutput[i] - targetOutput[0][i]
			currentError += diff * diff
		}
		currentError = math.Sqrt(currentError)
		if currentError < bestError {
			bestError = currentError
			for y := 0; y < inLayer.Height; y++ {
				copy(bestInput[y], currentInput[y])
			}
		}
		if currentError < 0.0001 {
			if n.Debug {
				fmt.Printf("Early stopping at iteration %d with error %.6f\n", iter, currentError)
			}
			break
		}
		if iter%100 == 0 && n.Debug {
			fmt.Printf("Iteration %d, Error: %.6f\n", iter, currentError)
		}
	}
	return bestInput
}

// ReverseUsingSimulatedAnnealing combines gradient descent with random exploration
func (n *Network[T]) ReverseUsingSimulatedAnnealing(targetOutput [][]float64, iterations int, initialTemp float64) [][]float64 {
	initScale, _, _, err := getInitScaleAndLr[T]()
	if err != nil {
		return n.ReversePropagate(targetOutput)
	}

	inLayer := n.Layers[n.InputLayer]
	currentInput := make([][]float64, inLayer.Height)
	bestInput := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		currentInput[y] = make([]float64, inLayer.Width)
		bestInput[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			currentInput[y][x] = float64(T(rand.Float64()-0.5) * initScale)
			bestInput[y][x] = currentInput[y][x]
		}
	}

	n.Forward(currentInput)
	output := n.GetOutput()
	currentError := 0.0
	for i := range output {
		diff := output[i] - targetOutput[0][i]
		currentError += diff * diff
	}
	currentError = math.Sqrt(currentError)
	bestError := currentError

	temperature := initialTemp
	coolingRate := 0.995

	for iter := 0; iter < iterations; iter++ {
		neighborInput := make([][]float64, inLayer.Height)
		for y := 0; y < inLayer.Height; y++ {
			neighborInput[y] = make([]float64, inLayer.Width)
			for x := 0; x < inLayer.Width; x++ {
				perturbation := (rand.Float64() - 0.5) * temperature
				neighborInput[y][x] = currentInput[y][x] + perturbation
				if neighborInput[y][x] > 5.0 {
					neighborInput[y][x] = 5.0
				} else if neighborInput[y][x] < -5.0 {
					neighborInput[y][x] = -5.0
				}
			}
		}
		n.Forward(neighborInput)
		neighborOutput := n.GetOutput()
		neighborError := 0.0
		for i := range neighborOutput {
			diff := neighborOutput[i] - targetOutput[0][i]
			neighborError += diff * diff
		}
		neighborError = math.Sqrt(neighborError)
		delta := neighborError - currentError
		if delta < 0 || rand.Float64() < math.Exp(-delta/temperature) {
			currentError = neighborError
			for y := 0; y < inLayer.Height; y++ {
				copy(currentInput[y], neighborInput[y])
			}
			if currentError < bestError {
				bestError = currentError
				for y := 0; y < inLayer.Height; y++ {
					copy(bestInput[y], currentInput[y])
				}
			}
		}
		temperature *= coolingRate
		if bestError < 0.0001 {
			if n.Debug {
				fmt.Printf("Early stopping at iteration %d with error %.6f\n", iter, bestError)
			}
			break
		}
		if iter%100 == 0 && n.Debug {
			fmt.Printf("Iteration %d, Error: %.6f, Temp: %.6f\n", iter, bestError, temperature)
		}
	}

	if n.Debug {
		fmt.Println("Final gradient refinement...")
	}
	refined := n.ReverseUsingAdam(targetOutput, 200)
	n.Forward(refined)
	refinedOutput := n.GetOutput()
	refinedError := 0.0
	for i := range refinedOutput {
		diff := refinedOutput[i] - targetOutput[0][i]
		refinedError += diff * diff
	}
	refinedError = math.Sqrt(refinedError)
	if refinedError < bestError {
		return refined
	}
	return bestInput
}

// AnalyzeInvertibility analyzes how invertible a network is likely to be
func (n *Network[T]) AnalyzeInvertibility() map[string]interface{} {
	analysis := make(map[string]interface{})
	reluCount := 0
	linearCount := 0
	totalNeurons := 0
	for l, layer := range n.Layers {
		if l == n.InputLayer {
			continue
		}
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				totalNeurons++
				switch layer.Neurons[y][x].Activation {
				case "relu":
					reluCount++
				case "linear", "identity":
					linearCount++
				}
			}
		}
	}
	inputSize := n.Layers[n.InputLayer].Width * n.Layers[n.InputLayer].Height
	outputSize := n.Layers[n.OutputLayer].Width * n.Layers[n.OutputLayer].Height
	dimensionalReduction := float64(outputSize) / float64(inputSize)
	score := 100.0
	reluPenalty := float64(reluCount) / float64(totalNeurons) * 30.0
	score -= reluPenalty
	if dimensionalReduction < 1.0 {
		reductionPenalty := (1.0 - dimensionalReduction) * 40.0
		score -= reductionPenalty
	}
	depthPenalty := float64(len(n.Layers)-2) * 5.0
	if depthPenalty > 30.0 {
		depthPenalty = 30.0
	}
	score -= depthPenalty
	linearBonus := float64(linearCount) / float64(totalNeurons) * 10.0
	score += linearBonus
	if score < 0 {
		score = 0
	}
	analysis["score"] = score
	analysis["relu_percentage"] = float64(reluCount) / float64(totalNeurons) * 100
	analysis["linear_percentage"] = float64(linearCount) / float64(totalNeurons) * 100
	analysis["dimensional_reduction"] = dimensionalReduction
	analysis["depth"] = len(n.Layers)
	analysis["input_size"] = inputSize
	analysis["output_size"] = outputSize
	recommendations := []string{}
	if reluCount > totalNeurons/2 {
		recommendations = append(recommendations, "Consider using LeakyReLU or ELU instead of ReLU")
	}
	if dimensionalReduction < 0.5 {
		recommendations = append(recommendations, "Severe dimensional reduction makes inversion difficult")
	}
	if len(n.Layers) > 5 {
		recommendations = append(recommendations, "Deep networks are harder to invert")
	}
	if score < 30 {
		recommendations = append(recommendations, "This network is likely very difficult to invert accurately")
	}
	analysis["recommendations"] = recommendations
	return analysis
}

// CreateInvertibleNetwork creates a network optimized for invertibility
func CreateInvertibleNetwork[T Numeric](
	inputSize, hiddenSize, outputSize int,
	numHiddenLayers int,
	linearOnly bool,
) *Network[T] {
	layers := []struct{ Width, Height int }{}
	activations := []string{}
	fullyConnected := []bool{}
	layers = append(layers, struct{ Width, Height int }{inputSize, 1})
	activations = append(activations, "linear")
	fullyConnected = append(fullyConnected, true)
	for i := 0; i < numHiddenLayers; i++ {
		layers = append(layers, struct{ Width, Height int }{hiddenSize, 1})
		if linearOnly {
			activations = append(activations, "linear")
		} else {
			activations = append(activations, "relu")
		}
		fullyConnected = append(fullyConnected, true)
	}
	layers = append(layers, struct{ Width, Height int }{outputSize, 1})
	activations = append(activations, "linear")
	fullyConnected = append(fullyConnected, true)
	network := NewNetwork[T](layers, activations, fullyConnected)
	if linearOnly {
		for l := 1; l < len(network.Layers); l++ {
			currLayer := network.Layers[l]
			for y := 0; y < currLayer.Height; y++ {
				for x := 0; x < currLayer.Width; x++ {
					neuron := currLayer.Neurons[y][x]
					for i, conn := range neuron.Inputs {
						if conn.SourceX == x && conn.SourceY == y {
							neuron.Inputs[i].Weight = T(1.0 + 0.01*(rand.Float64()-0.5))
						} else {
							neuron.Inputs[i].Weight = T(0.01 * (rand.Float64() - 0.5))
						}
					}
					neuron.Bias = T(0.01 * (rand.Float64() - 0.5))
				}
			}
		}
	}
	return network
}

// ReverseWithConstraints attempts to find inputs within specified constraints
func (n *Network[T]) ReverseWithConstraints(
	targetOutput [][]float64,
	minValues, maxValues []float64,
	iterations int,
) [][]float64 {
	_, _, _, err := getInitScaleAndLr[T]()
	if err != nil {
		return n.ReversePropagate(targetOutput)
	}

	inLayer := n.Layers[n.InputLayer]
	currentInput := make([][]float64, inLayer.Height)
	for y := 0; y < inLayer.Height; y++ {
		currentInput[y] = make([]float64, inLayer.Width)
		for x := 0; x < inLayer.Width; x++ {
			if x < len(minValues) && x < len(maxValues) {
				range_ := maxValues[x] - minValues[x]
				currentInput[y][x] = minValues[x] + rand.Float64()*range_
			} else {
				currentInput[y][x] = 0.0
			}
		}
	}
	result := n.ReverseUsingAdam(targetOutput, iterations)
	for y := 0; y < inLayer.Height; y++ {
		for x := 0; x < inLayer.Width; x++ {
			if x < len(minValues) && result[y][x] < minValues[x] {
				result[y][x] = minValues[x]
			}
			if x < len(maxValues) && result[y][x] > maxValues[x] {
				result[y][x] = maxValues[x]
			}
		}
	}
	return result
}

// GetLayerWeights returns the weight matrix and bias vector for layer l.
func (n *Network[T]) GetLayerWeights(l int) ([][]T, []T) {
	currLayer := n.Layers[l]
	prevLayer := n.Layers[l-1]
	wCurr, wPrev := currLayer.Width, prevLayer.Width
	W := make([][]T, wCurr)
	for j := 0; j < wCurr; j++ {
		W[j] = make([]T, wPrev)
		neuron := currLayer.Neurons[0][j]
		for _, conn := range neuron.Inputs {
			if conn.SourceLayer == l-1 {
				W[j][conn.SourceX] = conn.Weight
			}
		}
	}
	b := make([]T, wCurr)
	for j := 0; j < wCurr; j++ {
		b[j] = currLayer.Neurons[0][j].Bias
	}
	return W, b
}

// getInitScaleAndLr returns initialization scale, learning rate, and epsilon for type T, or an error if T is not floating-point.
func getInitScaleAndLr[T Numeric]() (T, T, T, error) {
	var initScale T
	var lr T
	var epsilon T
	var zero T

	switch any(zero).(type) {
	case float32:
		initScale = any(float32(0.1)).(T)
		lr = any(float32(0.01)).(T)
		epsilon = any(float32(1e-6)).(T)
	case float64:
		initScale = any(float64(0.1)).(T)
		lr = any(float64(0.01)).(T)
		epsilon = any(float64(1e-6)).(T)
	default:
		return zero, zero, zero, fmt.Errorf("requires floating-point type, got %T", zero)
	}

	return initScale, lr, epsilon, nil
}

// ReverseLayerByLayer reconstructs the input from the output, layer by layer.
func (n *Network[T]) ReverseLayerByLayer(output [][]float64) ([][]float64, error) {
	initScale, lr, epsilon, err := getInitScaleAndLr[T]()
	if err != nil {
		return nil, err
	}

	target := make([]T, len(output[0]))
	for i, v := range output[0] {
		target[i] = T(v)
	}
	for l := n.OutputLayer; l > 0; l-- {
		W, b := n.GetLayerWeights(l)
		var z []T
		if l == n.OutputLayer {
			z = target // Linear activation
		} else {
			activation := n.Layers[l].Neurons[0][0].Activation
			z = make([]T, len(target))
			for i, v := range target {
				z[i] = n.inverseActivation(v, activation)
			}
		}
		h := n.OptimizeH(W, b, z, initScale, lr, epsilon)
		target = h
	}
	reconstructed := make([][]float64, 1)
	reconstructed[0] = make([]float64, len(target))
	for i, v := range target {
		reconstructed[0][i] = float64(v)
	}
	return reconstructed, nil
}

// OptimizeH optimizes the input h to match the target z given weights W and biases b.
func (n *Network[T]) OptimizeH(W [][]T, b []T, z []T, initScale T, lr T, epsilon T) []T {
	wCurr, wPrev := len(W), len(W[0])
	h := make([]T, wPrev)
	for i := range h {
		randVal := rand.Float64() - 0.5
		h[i] = T(randVal) * initScale
	}
	maxIter := 500 // Increased iterations for better convergence
	var zero T
	for iter := 0; iter < maxIter; iter++ {
		Wh := make([]T, wCurr)
		for j := 0; j < wCurr; j++ {
			for k := 0; k < wPrev; k++ {
				Wh[j] += W[j][k] * h[k]
			}
			Wh[j] += b[j]
		}
		err := make([]T, wCurr)
		errorSum := zero
		for j := 0; j < wCurr; j++ {
			err[j] = Wh[j] - z[j]
			errorSum += err[j] * err[j]
		}
		if float64(errorSum) < float64(epsilon) {
			break // Early stopping if error is small
		}
		grad := make([]T, wPrev)
		for k := 0; k < wPrev; k++ {
			for j := 0; j < wCurr; j++ {
				grad[k] += W[j][k] * err[j]
			}
		}
		for k := 0; k < wPrev; k++ {
			h[k] -= lr * grad[k]
		}
	}
	return h
}
