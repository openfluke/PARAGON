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

	checkpointSize := n.Layers[checkpointLayer]
	outputSize := n.Layers[n.OutputLayer]

	// Create micro network: checkpoint → output (2 layers, replicating main network behavior)
	microLayerSizes := []struct{ Width, Height int }{
		{checkpointSize.Width, checkpointSize.Height}, // Layer 0: checkpoint input
		{outputSize.Width, outputSize.Height},         // Layer 1: output
	}

	microActivations := []string{
		n.Layers[checkpointLayer].Neurons[0][0].Activation, // checkpoint activation
		n.Layers[n.OutputLayer].Neurons[0][0].Activation,   // output activation
	}

	microFullyConnected := []bool{true, true}

	microNet, _ := NewNetwork[T](microLayerSizes, microActivations, microFullyConnected)
	microNet.Debug = false

	// Copy weights from original network: checkpoint → output becomes layer 0 → 1
	n.CopyWeightsBetweenNetworks(checkpointLayer, n.OutputLayer, microNet, 0, 1)

	return &MicroNetwork[T]{
		Network:       microNet,
		SourceLayers:  []int{checkpointLayer, n.OutputLayer},
		CheckpointIdx: 0, // Feed checkpoint data at layer 0
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
	checkpointLayer := mn.SourceLayers[0] // The checkpoint layer index

	// Check if micro network has improvement (3 layers vs original 2)
	if len(mn.Network.Layers) == 3 {
		// Insert new layer after checkpoint
		newLayerIdx := checkpointLayer + 1
		microHiddenLayer := mn.Network.Layers[1] // The new hidden layer

		// Add the new layer to the original network
		originalNet.AddLayer(newLayerIdx,
			microHiddenLayer.Width,
			microHiddenLayer.Height,
			microHiddenLayer.Neurons[0][0].Activation,
			true)

		// Copy weights: checkpoint→hidden (micro 0→1 becomes original checkpoint→newLayer)
		mn.Network.CopyWeightsBetweenNetworks(0, 1, originalNet, checkpointLayer, newLayerIdx)

		// Copy weights: hidden→output (micro 1→2 becomes original newLayer→output)
		originalOutputIdx := originalNet.OutputLayer // Output index has shifted
		mn.Network.CopyWeightsBetweenNetworks(1, 2, originalNet, newLayerIdx, originalOutputIdx)

	} else {
		// No improvement, just copy updated weights
		originalOutputIdx := originalNet.OutputLayer
		mn.Network.CopyWeightsBetweenNetworks(0, 1, originalNet, checkpointLayer, originalOutputIdx)
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

func (mn *MicroNetwork[T]) evaluatePerformance(checkpointActivations [][][]float64) float64 {
	totalScore := 0.0
	validCount := 0

	expectedW := mn.Network.Layers[0].Width
	expectedH := mn.Network.Layers[0].Height

	for _, activation := range checkpointActivations {
		// Skip if activation doesn't match expected input size
		if len(activation) != expectedH || len(activation[0]) != expectedW {
			continue
		}

		mn.Network.Forward(activation)
		output := mn.Network.GetOutput()

		// Score based on maximum confidence
		maxOutput := output[0]
		for _, val := range output[1:] {
			if val > maxOutput {
				maxOutput = val
			}
		}
		totalScore += maxOutput
		validCount++
	}

	if validCount == 0 {
		return 0.0
	}
	return totalScore / float64(validCount)
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
