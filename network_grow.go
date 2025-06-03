package paragon

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
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

	originalScore := n.Performance.Score
	type result struct {
		score float64
		micro *MicroNetwork[T]
	}
	results := make(chan result, numCandidates)

	var wg sync.WaitGroup
	jobs := make(chan int)

	// Start workers
	for t := 0; t < maxThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Recover from panic to prevent full crash
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("🔥 Recovered from panic in Grow() thread: %v\n", r)
				}
			}()

			// Clone the original network once per worker
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
				micro := workerNet.ExtractMicroNetwork(checkpointLayer)

				improved, success := micro.TryImprovement(
					testInputs,
					minWidth, maxWidth,
					minHeight, maxHeight,
					activationPool,
				)

				if !success {
					continue
				}

				targets := BuildTargetsFromLabels(expectedOutputs, n.Layers[n.OutputLayer].Width)
				improved.Network.Train(testInputs, targets, epochs, learningRate, false, clipUpper, clipLower)

				outputs := ExtractPredictedLabels(improved.Network, testInputs)
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

	// Apply result if better
	if bestMicro != nil && bestScore > originalScore {
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
		if err := bestMicro.ReattachToOriginal(newNet); err != nil {
			fmt.Println("Failed to reattach micro-network:", err)
			return false
		}

		newNet.EvaluateModel(expectedOutputs, ExtractPredictedLabels(newNet, testInputs))
		if newNet.Performance.Score > originalScore {
			*n = *newNet
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
	inW := net.Layers[0].Width
	inH := net.Layers[0].Height

	for i, in := range inputs {
		if len(in) != inH || len(in[0]) != inW {
			fmt.Printf("⚠️  Skipping input %d due to mismatched input shape: got %dx%d, want %dx%d\n",
				i, len(in[0]), len(in), inW, inH)
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

			net.Forward(in)
			raw := net.GetOutput()
			labels[i] = float64(ArgMax(raw))
		}()
	}

	return labels
}

func (mn *MicroNetwork[T]) TryImprovement(
	testInputs [][][]float64,
	minWidth int,
	maxWidth int,
	minHeight int,
	maxHeight int,
	activationPool []string,
) (*MicroNetwork[T], bool) {
	currentLayers := mn.Network.Layers

	// Extract existing activation functions
	inputAct := currentLayers[0].Neurons[0][0].Activation
	checkpointAct := currentLayers[1].Neurons[0][0].Activation
	outputAct := currentLayers[len(currentLayers)-1].Neurons[0][0].Activation

	// Randomize width, height, and activation for the new layer
	newWidth := rand.Intn(maxWidth-minWidth+1) + minWidth
	newHeight := rand.Intn(maxHeight-minHeight+1) + minHeight
	activation := activationPool[rand.Intn(len(activationPool))]

	// Define new architecture: input → checkpoint → new hidden → output
	improvedLayerSizes := []struct{ Width, Height int }{
		{currentLayers[0].Width, currentLayers[0].Height}, // Input
		{currentLayers[1].Width, currentLayers[1].Height}, // Checkpoint
		{newWidth, newHeight},                             // New hidden
		{currentLayers[2].Width, currentLayers[2].Height}, // Output
	}
	improvedActivations := []string{inputAct, checkpointAct, activation, outputAct}
	improvedFullyConnected := []bool{false, true, true, true}

	// Create improved network
	improvedNet := NewNetwork[T](improvedLayerSizes, improvedActivations, improvedFullyConnected)
	improvedNet.Debug = false

	// Copy weights from original micro to improved
	mn.Network.CopyWeightsBetweenNetworks(0, 1, improvedNet, 0, 1) // Input → Checkpoint
	mn.adaptOutputWeights(improvedNet)                             // Checkpoint → Output via new hidden

	// Package new micro network
	improvedMicro := &MicroNetwork[T]{
		Network:       improvedNet,
		SourceLayers:  mn.SourceLayers,
		CheckpointIdx: 1, // still layer 1 in micro
	}

	// Compare performance
	currentScore := mn.evaluatePerformance(testInputs)
	improvedScore := improvedMicro.evaluatePerformance(testInputs)

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
