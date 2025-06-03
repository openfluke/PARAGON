package paragon

import (
	"fmt"
)

// MicroNetwork represents a sub-network extracted from a larger network
type MicroNetwork[T Numeric] struct {
	Network       *Network[T]
	SourceLayers  []int // Original layer indices this micro network represents
	CheckpointIdx int   // Layer index where checkpoint was taken in micro network
}

// ExtractMicroNetwork creates a micro network from input to checkpoint layer to output
func (n *Network[T]) ExtractMicroNetwork(checkpointLayer int) *MicroNetwork[T] {
	if checkpointLayer <= n.InputLayer || checkpointLayer >= n.OutputLayer {
		panic(fmt.Sprintf("invalid checkpoint layer: %d (must be between %d and %d)",
			checkpointLayer, n.InputLayer+1, n.OutputLayer-1))
	}

	inputSize := n.Layers[n.InputLayer]
	checkpointSize := n.Layers[checkpointLayer]
	outputSize := n.Layers[n.OutputLayer]

	// Create micro network structure: input → checkpoint → output
	microLayerSizes := []struct{ Width, Height int }{
		{inputSize.Width, inputSize.Height},
		{checkpointSize.Width, checkpointSize.Height},
		{outputSize.Width, outputSize.Height},
	}

	microActivations := []string{
		n.Layers[n.InputLayer].Neurons[0][0].Activation,
		n.Layers[checkpointLayer].Neurons[0][0].Activation,
		n.Layers[n.OutputLayer].Neurons[0][0].Activation,
	}

	microFullyConnected := []bool{false, true, true}

	microNet := NewNetwork[T](microLayerSizes, microActivations, microFullyConnected)
	microNet.Debug = false

	// Copy weights: input → checkpoint (micro layer 0 → 1)
	n.CopyWeightsBetweenNetworks(n.InputLayer, checkpointLayer, microNet, 0, 1)

	// Copy weights: checkpoint → output (micro layer 1 → 2)
	n.CopyWeightsBetweenNetworks(checkpointLayer, n.OutputLayer, microNet, 1, 2)

	return &MicroNetwork[T]{
		Network:       microNet,
		SourceLayers:  []int{n.InputLayer, checkpointLayer, n.OutputLayer},
		CheckpointIdx: 1, // Layer 1 in micro network corresponds to checkpoint layer
	}
}

// VerifyThreeWayEquivalence performs the 3 critical verifications:
// 1. Main network full forward pass
// 2. Main network from checkpoint
// 3. Micro network from checkpoint
// Returns true if all three outputs match within tolerance
func (mn *MicroNetwork[T]) VerifyThreeWayEquivalence(originalNet *Network[T], input [][]float64, tolerance float64) (bool, [3][]float64) {
	checkpointLayer := mn.SourceLayers[1]

	// Verification 1: Main network full forward pass
	originalNet.Forward(input)
	fullForwardOutput := originalNet.GetOutput()
	checkpointState := originalNet.GetLayerState(checkpointLayer)

	// Verification 2: Main network from checkpoint
	originalNet.ForwardFromLayer(checkpointLayer, checkpointState)
	mainCheckpointOutput := originalNet.GetOutput()

	// Verification 3: Micro network from checkpoint
	mn.Network.ForwardFromLayer(mn.CheckpointIdx, checkpointState)
	microCheckpointOutput := mn.Network.GetOutput()

	// Store all outputs for return
	outputs := [3][]float64{
		fullForwardOutput,
		mainCheckpointOutput,
		microCheckpointOutput,
	}

	// Check if all three match within tolerance
	match1vs2 := outputsMatch(fullForwardOutput, mainCheckpointOutput, tolerance)
	match2vs3 := outputsMatch(mainCheckpointOutput, microCheckpointOutput, tolerance)
	match1vs3 := outputsMatch(fullForwardOutput, microCheckpointOutput, tolerance)

	allMatch := match1vs2 && match2vs3 && match1vs3
	return allMatch, outputs
}

// VerifyMicroNormalDiffers checks that micro network normal forward pass produces
// different outputs than checkpoint-based forward pass (as expected)
func (mn *MicroNetwork[T]) VerifyMicroNormalDiffers(input [][]float64, checkpointState [][]float64, tolerance float64) (bool, []float64, []float64) {
	// Normal forward pass
	mn.Network.Forward(input)
	normalOutput := mn.Network.GetOutput()

	// Checkpoint-based forward pass
	mn.Network.ForwardFromLayer(mn.CheckpointIdx, checkpointState)
	checkpointOutput := mn.Network.GetOutput()

	// They should be different
	isDifferent := !outputsMatch(normalOutput, checkpointOutput, tolerance)

	return isDifferent, normalOutput, checkpointOutput
}

// ReattachToOriginal updates the original network with improvements from micro network
func (mn *MicroNetwork[T]) ReattachToOriginal(originalNet *Network[T]) error {
	checkpointLayer := mn.SourceLayers[1]

	if len(mn.Network.Layers) > 3 {
		// Need to add new layer to original network
		newLayerIdx := checkpointLayer + 1
		newLayer := mn.Network.Layers[2] // The added hidden layer

		originalNet.AddLayer(newLayerIdx, newLayer.Width, newLayer.Height,
			newLayer.Neurons[0][0].Activation, true)

		// Copy weights
		mn.Network.CopyWeightsBetweenNetworks(1, 2, originalNet, checkpointLayer, newLayerIdx)
		mn.Network.CopyWeightsBetweenNetworks(2, 3, originalNet, newLayerIdx, originalNet.OutputLayer)
	} else {
		// Update existing weights
		mn.Network.CopyWeightsBetweenNetworks(0, 1, originalNet, mn.SourceLayers[0], checkpointLayer)
		mn.Network.CopyWeightsBetweenNetworks(1, 2, originalNet, checkpointLayer, originalNet.OutputLayer)
	}

	return nil
}

// Helper methods

func (srcNet *Network[T]) CopyWeightsBetweenNetworks(srcFromLayer, srcToLayer int,
	dstNet *Network[T], dstFromLayer, dstToLayer int) {

	srcLayer := srcNet.Layers[srcToLayer]
	dstLayer := &dstNet.Layers[dstToLayer]

	for y := 0; y < min(srcLayer.Height, dstLayer.Height); y++ {
		for x := 0; x < min(srcLayer.Width, dstLayer.Width); x++ {
			srcNeuron := srcLayer.Neurons[y][x]
			dstNeuron := dstLayer.Neurons[y][x]

			// Copy bias
			dstNeuron.Bias = srcNeuron.Bias

			// Copy weights
			maxConns := min(len(srcNeuron.Inputs), len(dstNeuron.Inputs))
			for i := 0; i < maxConns; i++ {
				dstNeuron.Inputs[i].Weight = srcNeuron.Inputs[i].Weight
				dstNeuron.Inputs[i].SourceLayer = dstFromLayer
				dstNeuron.Inputs[i].SourceX = srcNeuron.Inputs[i].SourceX
				dstNeuron.Inputs[i].SourceY = srcNeuron.Inputs[i].SourceY
			}
		}
	}
}

func (mn *MicroNetwork[T]) adaptOutputWeights(improvedNet *Network[T]) {
	origOutputLayer := mn.Network.Layers[2]       // Output layer in original micro
	improvedOutputLayer := &improvedNet.Layers[3] // Output layer in improved micro

	for y := 0; y < min(origOutputLayer.Height, improvedOutputLayer.Height); y++ {
		for x := 0; x < min(origOutputLayer.Width, improvedOutputLayer.Width); x++ {
			origNeuron := origOutputLayer.Neurons[y][x]
			improvedNeuron := improvedOutputLayer.Neurons[y][x]

			// Copy bias
			improvedNeuron.Bias = origNeuron.Bias

			// Adapt weights: take subset of original weights
			maxWeights := min(len(origNeuron.Inputs), len(improvedNeuron.Inputs))
			for i := 0; i < maxWeights; i++ {
				improvedNeuron.Inputs[i].Weight = origNeuron.Inputs[i].Weight
			}
		}
	}
}

func (mn *MicroNetwork[T]) evaluatePerformance(testInputs [][][]float64) float64 {
	totalScore := 0.0
	for _, input := range testInputs {
		mn.Network.Forward(input)
		output := mn.Network.GetOutput()

		// Score based on maximum confidence
		maxOutput := output[0]
		for _, val := range output[1:] {
			if val > maxOutput {
				maxOutput = val
			}
		}
		totalScore += maxOutput
	}
	return totalScore / float64(len(testInputs))
}

// Utility functions
func outputsMatch(output1, output2 []float64, tolerance float64) bool {
	if len(output1) != len(output2) {
		return false
	}
	for i := range output1 {
		if abs(output1[i]-output2[i]) > tolerance {
			return false
		}
	}
	return true
}

// NetworkSurgery performs complete micro network extraction, improvement, and reattachment
func (n *Network[T]) NetworkSurgery(
	checkpointLayer int,
	testInputs [][][]float64,
	tolerance float64,
	minWidth int,
	maxWidth int,
	minHeight int,
	maxHeight int,
	activationPool []string,
) (*MicroNetwork[T], error) {

	// Step 1: Extract micro network
	microNet := n.ExtractMicroNetwork(checkpointLayer)

	// Step 2: Verify equivalence
	if len(testInputs) > 0 {
		isEquivalent, _ := microNet.VerifyThreeWayEquivalence(n, testInputs[0], tolerance)
		if !isEquivalent {
			return nil, fmt.Errorf("micro network verification failed")
		}
	}

	// Step 3: Try improvements with parameters
	bestMicro, improved := microNet.TryImprovement(testInputs, minWidth, maxWidth, minHeight, maxHeight, activationPool)
	if !improved {
		fmt.Println("⚠️  No improvement found; reattaching original micro network")
	}

	// Step 4: Reattach
	if err := bestMicro.ReattachToOriginal(n); err != nil {
		return nil, fmt.Errorf("reattachment failed: %v", err)
	}

	return bestMicro, nil
}
