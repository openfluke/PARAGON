package paragon

import (
	"fmt"
	"math/rand"
)

func (n *Network[T]) Grow(
	checkpointLayer int,
	testInputs [][][]float64,
	expectedOutputs []float64,
	numCandidates int,
	epochs int,
	learningRate float64,
	tolerance float64,
	clipUpper T,
	clipLower T,
	minWidth int,
	maxWidth int,
	activationPool []string,
) bool {
	originalScore := n.Performance.Score
	var bestScore float64 = -1
	var bestMicro *MicroNetwork[T]

	for i := 0; i < numCandidates; i++ {
		// 1. Extract micro-network
		micro := n.ExtractMicroNetwork(checkpointLayer)

		// 2. Try improvement by adding a new layer
		improved, success := micro.TryImprovement(testInputs, minWidth, maxWidth, activationPool)
		if !success {
			continue
		}

		// 3. Train improved micro-network
		targets := BuildTargetsFromLabels(expectedOutputs, n.Layers[n.OutputLayer].Width)
		improved.Network.Train(testInputs, targets, epochs, learningRate, false, clipUpper, clipLower)

		// 4. Evaluate ADHD score
		outputs := ExtractPredictedLabels(improved.Network, testInputs)
		improved.Network.EvaluateModel(expectedOutputs, outputs)
		score := improved.Network.Performance.Score

		if score > bestScore {
			bestScore = score
			bestMicro = improved
		}
	}

	if bestMicro != nil && bestScore > originalScore {
		// Deep copy original network
		newNet := &Network[T]{}
		data, err := n.MarshalJSONModel()
		if err != nil {
			fmt.Println("Failed to marshal network:", err)
			return false
		}
		if err := newNet.UnmarshalJSONModel(data); err != nil {
			fmt.Println("Failed to unmarshal into network copy:", err)
			return false
		}

		// Reattach the improved micro-network
		if err := bestMicro.ReattachToOriginal(newNet); err != nil {
			fmt.Println("Failed to reattach micro-network:", err)
			return false
		}

		// Re-evaluate and accept if better
		newNet.EvaluateModel(expectedOutputs, ExtractPredictedLabels(newNet, testInputs))
		if newNet.Performance.Score > originalScore {
			*n = *newNet // Apply improvement
			fmt.Printf("✅ Network improved from %.2f → %.2f via Grow()\n", originalScore, newNet.Performance.Score)
			return true
		}
	}

	fmt.Println("⚠️ Grow() found no candidate that outperformed the current network")
	return false
}

func BuildTargetsFromLabels(labels []float64, numClasses int) [][][]float64 {
	targets := make([][][]float64, len(labels))
	for i, label := range labels {
		vec := make([]float64, numClasses)
		class := int(label)
		if class >= 0 && class < numClasses {
			vec[class] = 1.0
		}
		targets[i] = [][]float64{vec}
	}
	return targets
}

func ExtractPredictedLabels[T Numeric](net *Network[T], inputs [][][]float64) []float64 {
	labels := make([]float64, len(inputs))
	for i, in := range inputs {
		net.Forward(in)
		raw := net.GetOutput()
		labels[i] = float64(ArgMax(raw))
	}
	return labels
}

func (mn *MicroNetwork[T]) TryImprovement(testInputs [][][]float64, minWidth int, maxWidth int, activationPool []string) (*MicroNetwork[T], bool) {
	currentLayers := mn.Network.Layers

	// Read original activations
	inputAct := currentLayers[0].Neurons[0][0].Activation
	checkpointAct := currentLayers[1].Neurons[0][0].Activation
	outputAct := currentLayers[len(currentLayers)-1].Neurons[0][0].Activation

	// Dynamically insert a new hidden layer with random width and activation
	newWidth := rand.Intn(maxWidth-minWidth+1) + minWidth
	activation := activationPool[rand.Intn(len(activationPool))]

	improvedLayerSizes := []struct{ Width, Height int }{
		{currentLayers[0].Width, currentLayers[0].Height}, // Input
		{currentLayers[1].Width, currentLayers[1].Height}, // Checkpoint
		{newWidth, 1}, // New hidden
		{currentLayers[2].Width, currentLayers[2].Height}, // Output
	}
	improvedActivations := []string{inputAct, checkpointAct, activation, outputAct}
	improvedFullyConnected := []bool{false, true, true, true}

	// Build improved net
	improvedNet := NewNetwork[T](improvedLayerSizes, improvedActivations, improvedFullyConnected)
	improvedNet.Debug = false

	// Copy weights
	mn.Network.CopyWeightsBetweenNetworks(0, 1, improvedNet, 0, 1) // Input → Checkpoint
	mn.adaptOutputWeights(improvedNet)                             // Checkpoint → Output (via new hidden)

	// Package micro network
	improvedMicro := &MicroNetwork[T]{
		Network:       improvedNet,
		SourceLayers:  mn.SourceLayers,
		CheckpointIdx: 1,
	}

	// Evaluate improvement
	currentScore := mn.evaluatePerformance(testInputs)
	improvedScore := improvedMicro.evaluatePerformance(testInputs)

	if improvedScore > currentScore {
		return improvedMicro, true
	}
	return mn, false
}
