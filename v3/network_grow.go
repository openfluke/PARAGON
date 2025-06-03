package paragon

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"
)

type GrowthLog struct {
	LayerIndex  int     `json:"layer_index"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	Activation  string  `json:"activation"`
	ScoreBefore float64 `json:"score_before"`
	ScoreAfter  float64 `json:"score_after"`
	Timestamp   string  `json:"timestamp"`
}

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
	minHeight int,
	maxHeight int,
	activationPool []string,
	maxThreads int,
) bool {
	// FIXED: Calculate baseline score on THIS batch, not global score
	baselineScore := n.calculateBatchScore(testInputs, expectedOutputs)
	fmt.Printf("🎯 Baseline score on this batch: %.2f\n", baselineScore)

	type result struct {
		score float64
		micro *MicroNetwork[T]
	}
	results := make(chan result, numCandidates)

	var wg sync.WaitGroup
	jobs := make(chan int)

	// Extract checkpoint activations from the original inputs
	checkpointActivations := ExtractActivationsAtLayer(n, checkpointLayer, testInputs)

	// Start workers
	for t := 0; t < maxThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("🔥 Recovered from panic in Grow() thread: %v\n", r)
				}
			}()

			workerNet := &Network[T]{}
			data, err := n.MarshalJSONModel()
			if err != nil {
				fmt.Println("❌ Failed to marshal original network for cloning:", err)
				return
			}
			if err := workerNet.UnmarshalJSONModel(data); err != nil {
				fmt.Println("❌ Failed to unmarshal cloned network:", err)
				return
			}

			for range jobs {
				// Extract micro network from checkpoint to output
				micro := workerNet.ExtractMicroNetwork(checkpointLayer)

				// Try to improve the micro network by adding/modifying hidden layer
				improved, success := micro.TryImprovement(
					checkpointActivations, // Pass checkpoint activations, not original inputs
					minWidth, maxWidth,
					minHeight, maxHeight,
					activationPool,
				)
				if !success {
					continue
				}

				// Train the improved micro network on checkpoint activations
				targets := BuildTargetsFromLabels(expectedOutputs, n.Layers[n.OutputLayer].Width)
				improved.Network.Train(checkpointActivations, targets, epochs, learningRate, false, clipUpper, clipLower)

				// Evaluate the micro network using checkpoint activations
				outputs := ExtractPredictedLabels(improved.Network, checkpointActivations)
				improved.Network.EvaluateModel(expectedOutputs, outputs)

				results <- result{
					score: improved.Network.Performance.Score,
					micro: improved,
				}
			}
		}()
	}

	// Feed jobs
	go func() {
		for i := 0; i < numCandidates; i++ {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	// Select best candidate
	var bestScore float64 = -1
	var bestMicro *MicroNetwork[T]
	for r := range results {
		if r.score > bestScore {
			bestScore = r.score
			bestMicro = r.micro
		}
	}

	fmt.Printf("🏆 Best micro score: %.2f vs baseline: %.2f\n", bestScore, baselineScore)

	// FIXED: Compare against baseline score on this batch, with small improvement threshold
	improvementThreshold := 1.0 // Require at least 1% improvement
	if bestMicro != nil && bestScore > baselineScore+improvementThreshold {
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

		// Reattach the improved micro network to create the new full network
		if err := bestMicro.ReattachToOriginal(newNet); err != nil {
			fmt.Println("Failed to reattach micro-network:", err)
			return false
		}

		// Now evaluate the NEW FULL NETWORK on this batch
		newBatchScore := newNet.calculateBatchScore(testInputs, expectedOutputs)
		fmt.Printf("🔍 New network score on batch: %.2f\n", newBatchScore)

		// FIXED: Compare new full network against baseline on same batch
		if newBatchScore > baselineScore+improvementThreshold {
			// Copy the existing growth history from the original network first
			if n.GrowthHistory != nil {
				newNet.GrowthHistory = make([]GrowthLog, len(n.GrowthHistory))
				copy(newNet.GrowthHistory, n.GrowthHistory)
			} else {
				newNet.GrowthHistory = []GrowthLog{}
			}

			// Find the actual new layer that was added
			newLayerIdx := checkpointLayer + 1

			// Ensure we're logging the correct layer index
			if newLayerIdx < len(newNet.Layers) {
				// Always append the new growth log
				newNet.GrowthHistory = append(newNet.GrowthHistory, GrowthLog{
					LayerIndex:  newLayerIdx,
					Width:       newNet.Layers[newLayerIdx].Width,
					Height:      newNet.Layers[newLayerIdx].Height,
					Activation:  newNet.Layers[newLayerIdx].Neurons[0][0].Activation,
					ScoreBefore: baselineScore,
					ScoreAfter:  newBatchScore,
					Timestamp:   time.Now().Format(time.RFC3339),
				})

				fmt.Printf("📝 Logged growth #%d: Layer %d (%dx%d, %s)\n",
					len(newNet.GrowthHistory),
					newLayerIdx,
					newNet.Layers[newLayerIdx].Width,
					newNet.Layers[newLayerIdx].Height,
					newNet.Layers[newLayerIdx].Neurons[0][0].Activation)
			}

			*n = *newNet
			fmt.Printf("✅ Network improved from %.2f → %.2f via Grow()\n", baselineScore, newBatchScore)
			return true
		} else {
			fmt.Printf("⚠️ Full network score %.2f not better than baseline %.2f\n", newBatchScore, baselineScore)
		}
	} else {
		fmt.Printf("⚠️ Best micro score %.2f not better than baseline %.2f + threshold %.2f\n",
			bestScore, baselineScore, improvementThreshold)
	}

	fmt.Println("⚠️ Grow() found no candidate that outperformed the current network")
	return false
}

// Helper function to calculate network score on a specific batch
func (n *Network[T]) calculateBatchScore(inputs [][][]float64, expectedOutputs []float64) float64 {
	expectedLabels := make([]float64, len(inputs))
	actualLabels := make([]float64, len(inputs))

	for i, input := range inputs {
		n.Forward(input)
		output := n.GetOutput()
		actualLabels[i] = float64(ArgMax(output))
		expectedLabels[i] = expectedOutputs[i]
	}

	// Calculate accuracy as percentage
	correct := 0
	for i := range expectedLabels {
		if expectedLabels[i] == actualLabels[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(expectedLabels)) * 100.0
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

func ExtractPredictedLabels[T Numeric](net *Network[T], checkpointActivations [][][]float64) []float64 {
	labels := make([]float64, len(checkpointActivations))

	// The micro network expects checkpoint activations as input
	expectedW := net.Layers[0].Width
	expectedH := net.Layers[0].Height

	for i, activation := range checkpointActivations {
		if len(activation) != expectedH || len(activation[0]) != expectedW {
			fmt.Printf("⚠️  Skipping input %d due to mismatched activation shape: got %dx%d, want %dx%d\n",
				i, len(activation[0]), len(activation), expectedW, expectedH)
			labels[i] = -1
			continue
		}

		// Recover from per-input panic
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("🔥 Recovered from panic in input %d Forward(): %v\n", i, r)
					labels[i] = -1
				}
			}()

			net.Forward(activation) // Feed checkpoint activation directly
			raw := net.GetOutput()
			labels[i] = float64(ArgMax(raw))
		}()
	}

	return labels
}

func (mn *MicroNetwork[T]) TryImprovement(
	checkpointActivations [][][]float64,
	minWidth int,
	maxWidth int,
	minHeight int,
	maxHeight int,
	activationPool []string,
) (*MicroNetwork[T], bool) {
	currentLayers := mn.Network.Layers

	// Validate micro-network has exactly 2 layers: checkpoint → output
	if len(currentLayers) != 2 {
		fmt.Printf("❌ TryImprovement: expected 2 layers (checkpoint → output), got %d\n", len(currentLayers))
		return mn, false
	}

	checkpointAct := currentLayers[0].Neurons[0][0].Activation
	outputAct := currentLayers[1].Neurons[0][0].Activation

	// Generate new hidden layer dimensions
	newWidth := rand.Intn(maxWidth-minWidth+1) + minWidth
	newHeight := rand.Intn(maxHeight-minHeight+1) + minHeight
	newAct := activationPool[rand.Intn(len(activationPool))]

	// Create improved network: checkpoint → new hidden → output (3 layers)
	improvedLayerSizes := []struct{ Width, Height int }{
		{currentLayers[0].Width, currentLayers[0].Height}, // checkpoint input (unchanged)
		{newWidth, newHeight},                             // NEW hidden layer
		{currentLayers[1].Width, currentLayers[1].Height}, // output (unchanged)
	}
	improvedActivations := []string{checkpointAct, newAct, outputAct}
	improvedFullyConnected := []bool{true, true, true}

	improvedNet := NewNetwork[T](improvedLayerSizes, improvedActivations, improvedFullyConnected)
	improvedNet.Debug = false

	// Copy output weights from original micro (0→1) to improved (1→2)
	mn.adaptOutputWeightsForNewLayer(improvedNet)

	improvedMicro := &MicroNetwork[T]{
		Network:       improvedNet,
		SourceLayers:  mn.SourceLayers,
		CheckpointIdx: 0,
	}

	// Evaluate both networks on checkpoint activations
	currentScore := mn.evaluatePerformance(checkpointActivations)
	improvedScore := improvedMicro.evaluatePerformance(checkpointActivations)

	if improvedScore > currentScore {
		return improvedMicro, true
	}
	return mn, false
}

func (n *Network[T]) PrintGrowthHistory() {
	if len(n.GrowthHistory) == 0 {
		fmt.Println("📭 No growth history logged.")
		return
	}
	fmt.Println("🌱 Growth History Log:")
	for _, g := range n.GrowthHistory {
		fmt.Printf("  ➕ Layer %d: %dx%d (%s), Score %.2f → %.2f [%s]\n",
			g.LayerIndex, g.Width, g.Height, g.Activation,
			g.ScoreBefore, g.ScoreAfter, g.Timestamp)
	}
}

func (n *Network[T]) SaveGrowthLogJSON(path string) error {
	data, err := json.MarshalIndent(n.GrowthHistory, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func ExtractActivationsAtLayer[T Numeric](net *Network[T], layerIndex int, inputs [][][]float64) [][][]float64 {
	outputs := make([][][]float64, len(inputs))

	for i, input := range inputs {
		net.Forward(input)

		layer := net.Layers[layerIndex]
		activation := make([][]float64, layer.Height)
		for y := 0; y < layer.Height; y++ {
			activation[y] = make([]float64, layer.Width)
			for x := 0; x < layer.Width; x++ {
				activation[y][x] = float64(layer.Neurons[y][x].Value)
			}
		}
		outputs[i] = activation
	}
	return outputs
}

func (mn *MicroNetwork[T]) adaptOutputWeightsForInsertedLayer(improvedNet *Network[T]) {
	// From original: checkpoint → output = 0→1
	// To improved: new hidden → output = 1→2
	origOutput := mn.Network.Layers[1]
	newOutput := &improvedNet.Layers[2]

	for y := 0; y < min(origOutput.Height, newOutput.Height); y++ {
		for x := 0; x < min(origOutput.Width, newOutput.Width); x++ {
			src := origOutput.Neurons[y][x]
			dst := newOutput.Neurons[y][x]

			dst.Bias = src.Bias
			maxW := min(len(src.Inputs), len(dst.Inputs))
			for i := 0; i < maxW; i++ {
				dst.Inputs[i].Weight = src.Inputs[i].Weight
			}
		}
	}
}

func (mn *MicroNetwork[T]) adaptOutputWeightsForNewHidden(improvedNet *Network[T]) {
	// Copy from original micro layer 2 (output) to improved layer 2 (output)
	origOutput := mn.Network.Layers[2]
	newOutput := &improvedNet.Layers[2]

	for y := 0; y < min(origOutput.Height, newOutput.Height); y++ {
		for x := 0; x < min(origOutput.Width, newOutput.Width); x++ {
			src := origOutput.Neurons[y][x]
			dst := newOutput.Neurons[y][x]

			dst.Bias = src.Bias

			// Copy as many weights as possible from the original connections
			maxW := min(len(src.Inputs), len(dst.Inputs))
			for i := 0; i < maxW; i++ {
				dst.Inputs[i].Weight = src.Inputs[i].Weight
			}
		}
	}
}

func (mn *MicroNetwork[T]) adaptOutputWeightsForNewLayer(improvedNet *Network[T]) {
	// Copy from original micro layer 1 (output) to improved layer 2 (output)
	origOutput := mn.Network.Layers[1]  // Original output layer
	newOutput := &improvedNet.Layers[2] // New output layer (shifted due to insertion)

	for y := 0; y < min(origOutput.Height, newOutput.Height); y++ {
		for x := 0; x < min(origOutput.Width, newOutput.Width); x++ {
			src := origOutput.Neurons[y][x]
			dst := newOutput.Neurons[y][x]

			dst.Bias = src.Bias

			// Copy weights (the new hidden layer will have random weights initially)
			maxW := min(len(src.Inputs), len(dst.Inputs))
			for i := 0; i < maxW; i++ {
				dst.Inputs[i].Weight = src.Inputs[i].Weight
			}
		}
	}
}
