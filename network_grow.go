package paragon

import (
	"fmt"
	"math/rand"
	"time"
)

// GrowConfig defines parameters for the network growth process
type GrowConfig struct {
	// Core parameters
	BatchSize            int     // Number of checkpoints per batch (default: 64)
	MicroNetCount        int     // Number of micro networks per batch (default: 16)
	MinADHDScore         float64 // Minimum ADHD score threshold to trigger growth (default: 80.0)
	ImprovementThreshold float64 // Minimum ADHD improvement required to accept changes (default: 5.0)

	// Training parameters
	TrainingEpochs int     // Epochs to train micro networks (default: 1)
	LearningRate   float64 // Learning rate for micro network training (default: 0.02)
	Tolerance      float64 // Tolerance for equivalence checking (default: 1e-6)

	// Layer growth parameters
	NewLayerWidth      int    // Width of new layer to add (default: 5)
	NewLayerHeight     int    // Height of new layer to add (default: 1)
	NewLayerActivation string // Activation function for new layer (default: "relu")

	// Performance tuning
	MaxGrowthAttempts int     // Maximum growth attempts per call (default: 1)
	ValidationSplit   float64 // Fraction of data for validation (default: 0.2)

	// Debug options
	Debug              bool // Enable debug output
	SaveCheckpoints    bool // Save checkpoint data for analysis
	UseGPUForMicroNets bool
}

// DefaultGrowConfig returns a GrowConfig with sensible defaults for ADHD-based growth
func DefaultGrowConfig() *GrowConfig {
	return &GrowConfig{
		BatchSize:            64,   // 64 samples as requested
		MicroNetCount:        16,   // Fewer micro nets for faster processing
		MinADHDScore:         80.0, // ADHD score threshold
		ImprovementThreshold: 5.0,  // ADHD improvement threshold
		TrainingEpochs:       1,    // 1 epoch as requested
		LearningRate:         0.02,
		Tolerance:            1e-6,
		NewLayerWidth:        5,
		NewLayerHeight:       1,
		NewLayerActivation:   "relu",
		MaxGrowthAttempts:    1,
		ValidationSplit:      0.2,
		Debug:                true,
		SaveCheckpoints:      false,
		UseGPUForMicroNets:   true, // New: enable GPU by default

	}
}

// GrowthResult contains the results of a growth attempt with ADHD metrics
type GrowthResult struct {
	Success            bool
	OriginalADHDScore  float64
	ImprovedADHDScore  float64
	ADHDImprovement    float64
	LayersAdded        int
	ProcessingTime     time.Duration
	CheckpointLayer    int
	MicroNetsProcessed int
	BestMicroNetScore  float64
	ADHDBreakdown      *ADHDPerformance
}

// CheckpointBatch represents a batch of checkpoints for processing
type CheckpointBatch struct {
	States    [][][]float64 // Checkpoint states
	Inputs    [][][]float64 // Original inputs that created these states
	Targets   [][][]float64 // Expected outputs
	Layer     int           // Layer index where checkpoints were taken
	ADHDScore float64       // Current ADHD score on these samples
}

// MicroNetCandidate represents a micro network candidate for improvement
type MicroNetCandidate[T Numeric] struct {
	MicroNet        *MicroNetwork[T]
	ADHDScore       float64
	ADHDImprovement float64
	TrainingLoss    float64
}

// Grow attempts to improve the network by growing layers where ADHD performance is poor
func (n *Network[T]) Grow(inputs, targets [][][]float64, config *GrowConfig) (*GrowthResult, error) {
	if config == nil {
		config = DefaultGrowConfig()
	}

	startTime := time.Now()

	if config.Debug {
		fmt.Println("🌱 Starting ADHD-based network growth process...")
		fmt.Printf("📊 Processing %d samples with config: batch_size=%d, micro_nets=%d\n",
			len(inputs), config.BatchSize, config.MicroNetCount)
	}

	result := &GrowthResult{
		Success:        false,
		ProcessingTime: 0,
		LayersAdded:    0,
	}

	// Step 1: Evaluate current ADHD performance
	originalADHDScore := n.EvaluateFullScore(inputs, targets)
	result.OriginalADHDScore = originalADHDScore
	fmt.Println("OriginalADHDScore: ", result.OriginalADHDScore)

	if config.Debug {
		fmt.Printf("📈 Original network ADHD score: %.4f\n", originalADHDScore)
	}

	if originalADHDScore >= config.MinADHDScore {
		if config.Debug {
			fmt.Printf("✅ Network already meets ADHD threshold (%.4f >= %.4f)\n",
				originalADHDScore, config.MinADHDScore)
		}
		result.ProcessingTime = time.Since(startTime)
		return result, nil
	}

	// Step 2: Find best checkpoint layer for growth using ADHD analysis
	bestLayer, poorSamples := n.findBestGrowthPointADHD(inputs, targets, config)
	if bestLayer == -1 {
		if config.Debug {
			fmt.Println("❌ No suitable growth point found via ADHD analysis")
		}
		result.ProcessingTime = time.Since(startTime)
		return result, fmt.Errorf("no suitable growth point found")
	}

	result.CheckpointLayer = bestLayer

	if config.Debug {
		fmt.Printf("🎯 Selected layer %d for growth (found %d ADHD-poor samples)\n",
			bestLayer, len(poorSamples))
	}

	// Step 3: Create exactly 64 checkpoint samples as requested
	batch := n.createADHDCheckpointBatch(inputs, targets, poorSamples, bestLayer, config)

	if config.Debug {
		fmt.Printf("📋 Created checkpoint batch with %d samples from layer %d\n",
			len(batch.States), batch.Layer)
		fmt.Printf("📊 Batch ADHD score: %.4f\n", batch.ADHDScore)
	}

	// Step 4: Process micro networks with ADHD evaluation
	bestADHDImprovement := 0.0
	var bestMicroNet *MicroNetwork[T]
	totalMicroNets := 0

	for attempt := 0; attempt < config.MaxGrowthAttempts; attempt++ {
		if config.Debug {
			fmt.Printf("\n🔄 Growth attempt %d/%d\n", attempt+1, config.MaxGrowthAttempts)
		}

		// Process micro networks for this batch
		candidates := n.processMicroNetBatchADHD(batch, config)
		totalMicroNets += len(candidates)

		// Find best candidate based on ADHD improvement
		for _, candidate := range candidates {
			if candidate.ADHDImprovement > bestADHDImprovement {
				bestADHDImprovement = candidate.ADHDImprovement
				bestMicroNet = candidate.MicroNet
				result.BestMicroNetScore = candidate.ADHDScore
			}
		}

		if config.Debug {
			fmt.Printf("📊 Batch processed: %d candidates, best ADHD improvement: %.4f\n",
				len(candidates), bestADHDImprovement)
		}

		// Check if we have sufficient ADHD improvement
		if bestADHDImprovement >= config.ImprovementThreshold {
			break
		}
	}

	result.MicroNetsProcessed = totalMicroNets

	// Step 5: Apply best improvement if found and verify on full model
	if bestMicroNet != nil && bestADHDImprovement >= config.ImprovementThreshold {
		if config.Debug {
			fmt.Printf("🚀 Applying best ADHD improvement (%.4f improvement)\n", bestADHDImprovement)
		}

		// Store original network state in case we need to revert
		originalLayerCount := len(n.Layers)

		err := bestMicroNet.ReattachToOriginal(n)
		if err != nil {
			result.ProcessingTime = time.Since(startTime)
			return result, fmt.Errorf("failed to reattach improved micro network: %v", err)
		}

		// Evaluate improved network with full ADHD analysis
		improvedADHDScore := n.EvaluateFullScore(inputs, targets)
		result.ImprovedADHDScore = improvedADHDScore
		result.ADHDImprovement = improvedADHDScore - originalADHDScore

		// Only consider success if full model actually improved
		if result.ADHDImprovement > 0 {
			result.Success = true
			result.LayersAdded = len(n.Layers) - originalLayerCount
			result.ADHDBreakdown = n.Performance

			if config.Debug {
				fmt.Printf("✅ Growth successful! ADHD Score: %.4f → %.4f (%.4f improvement)\n",
					originalADHDScore, improvedADHDScore, result.ADHDImprovement)
				fmt.Printf("🏗️  Added %d layers to the network\n", result.LayersAdded)
				fmt.Println("🎉 Full model ADHD improvement confirmed!")
			}
		} else {
			// Full model got worse - this indicates a problem with reattachment
			result.Success = false
			result.LayersAdded = len(n.Layers) - originalLayerCount // Still count layers added
			result.ADHDBreakdown = n.Performance

			if config.Debug {
				fmt.Printf("❌ Growth failed! ADHD Score: %.4f → %.4f (%.4f decline)\n",
					originalADHDScore, improvedADHDScore, result.ADHDImprovement)
				fmt.Printf("🏗️  Added %d layers but performance declined\n", result.LayersAdded)
				fmt.Println("⚠️  Micro net improved but full model declined - reattachment issue detected")

				// Could add revert logic here if desired
				// For now, we'll keep the changes for analysis
			}
		}
	} else {
		if config.Debug {
			fmt.Printf("❌ No sufficient ADHD improvement found (best: %.4f, required: %.4f)\n",
				bestADHDImprovement, config.ImprovementThreshold)
		}
	}

	result.ProcessingTime = time.Since(startTime)
	return result, nil
}

// evaluateADHDScore evaluates network performance using ADHD metrics
func (n *Network[T]) evaluateADHDScore(inputs, targets [][][]float64) float64 {
	expectedOutputs := make([]float64, len(inputs))
	actualOutputs := make([]float64, len(inputs))

	for i := range inputs {
		n.Forward(inputs[i])
		output := n.GetOutput()

		// Get expected and actual class predictions
		expectedOutputs[i] = float64(ArgMax(targets[i][0]))
		actualOutputs[i] = float64(ArgMax(output))
	}

	// Use ADHD evaluation
	n.EvaluateModel(expectedOutputs, actualOutputs)
	return n.ComputeFinalScore()
}

// evaluateCheckpointADHDScore evaluates checkpoint performance using ADHD metrics
func (n *Network[T]) evaluateCheckpointADHDScore(checkpoints [][][]float64, targets [][][]float64, checkpointLayer int) float64 {
	expectedOutputs := make([]float64, len(targets))
	for i := range targets {
		expectedOutputs[i] = float64(ArgMax(targets[i][0]))
	}

	// Debug: Check if we have valid data
	if len(checkpoints) == 0 || len(expectedOutputs) == 0 {
		return 0.0
	}

	n.EvaluateFromCheckpoint(checkpoints, expectedOutputs, checkpointLayer)
	score := n.ComputeFinalScore()

	// Debug: Ensure we got a valid score
	if score == 0.0 {
		// Fallback: evaluate manually if EvaluateFromCheckpoint failed
		actualOutputs := make([]float64, len(checkpoints))
		for i := range checkpoints {
			n.ForwardFromLayer(checkpointLayer, checkpoints[i])
			output := n.GetOutput()
			actualOutputs[i] = float64(ArgMax(output))
		}
		n.EvaluateModel(expectedOutputs, actualOutputs)
		score = n.ComputeFinalScore()
	}

	return score
}

// findBestGrowthPointADHD identifies the best layer using ADHD analysis
func (n *Network[T]) findBestGrowthPointADHD(inputs, targets [][][]float64, config *GrowConfig) (int, []int) {
	var poorSamples []int
	bestLayer := -1
	lowestADHDScore := 100.0 // Start with perfect score

	// Test each potential checkpoint layer
	for layer := 1; layer < n.OutputLayer; layer++ {
		currentPoorSamples := n.identifyADHDPoorSamples(inputs, targets, layer, config)

		if len(currentPoorSamples) > 0 {
			// Create temporary batch to evaluate ADHD score for this layer
			tempBatch := n.createADHDCheckpointBatch(inputs, targets, currentPoorSamples, layer, config)

			if tempBatch.ADHDScore < lowestADHDScore {
				lowestADHDScore = tempBatch.ADHDScore
				bestLayer = layer
				poorSamples = currentPoorSamples
			}
		}
	}

	return bestLayer, poorSamples
}

// identifyADHDPoorSamples finds samples with poor ADHD performance from checkpoint
func (n *Network[T]) identifyADHDPoorSamples(inputs, targets [][][]float64, checkpointLayer int, config *GrowConfig) []int {
	var poorSamples []int

	for i := range inputs {
		// Forward pass to get checkpoint state
		n.Forward(inputs[i])
		checkpointState := n.GetLayerState(checkpointLayer)

		// Forward from checkpoint
		n.ForwardFromLayer(checkpointLayer, checkpointState)
		output := n.GetOutput()

		// Evaluate this single sample with ADHD
		expected := float64(ArgMax(targets[i][0]))
		actual := float64(ArgMax(output))

		result := n.EvaluatePrediction(expected, actual)

		// Consider sample "poor" if deviation is medium-high or in bad buckets
		// Lowered threshold to catch more samples for improvement
		if result.Deviation > 25.0 || result.Bucket == "100%+" || result.Bucket == "50-100%" || result.Bucket == "40-50%" {
			poorSamples = append(poorSamples, i)
		}
	}

	return poorSamples
}

// createADHDCheckpointBatch creates a batch of exactly 64 samples with ADHD evaluation
func (n *Network[T]) createADHDCheckpointBatch(inputs, targets [][][]float64, poorSamples []int, checkpointLayer int, config *GrowConfig) *CheckpointBatch {
	batchSize := config.BatchSize // Use exactly 64 samples
	if len(poorSamples) < batchSize {
		// If not enough poor samples, repeat some
		for len(poorSamples) < batchSize {
			poorSamples = append(poorSamples, poorSamples...)
		}
	}

	// Take exactly batchSize samples
	if len(poorSamples) > batchSize {
		indices := rand.Perm(len(poorSamples))[:batchSize]
		selectedPoor := make([]int, batchSize)
		for i, idx := range indices {
			selectedPoor[i] = poorSamples[idx]
		}
		poorSamples = selectedPoor
	}

	batch := &CheckpointBatch{
		States:  make([][][]float64, batchSize),
		Inputs:  make([][][]float64, batchSize),
		Targets: make([][][]float64, batchSize),
		Layer:   checkpointLayer,
	}

	checkpoints := make([][][]float64, batchSize)

	for i, sampleIdx := range poorSamples {
		// Get real checkpoint state by running forward pass
		n.Forward(inputs[sampleIdx])
		batch.States[i] = n.GetLayerState(checkpointLayer)
		checkpoints[i] = batch.States[i]

		// Use real input and target data
		batch.Inputs[i] = inputs[sampleIdx]
		batch.Targets[i] = targets[sampleIdx]
	}

	// Evaluate batch ADHD score
	batch.ADHDScore = n.evaluateCheckpointADHDScore(checkpoints, batch.Targets, checkpointLayer)

	// Debug information
	if config.Debug {
		fmt.Printf("   🔍 Debug: Batch created with %d samples\n", len(batch.States))
		fmt.Printf("   🔍 Debug: First checkpoint state size: %dx%d\n",
			len(batch.States[0]), len(batch.States[0][0]))
		fmt.Printf("   🔍 Debug: Evaluating from layer %d\n", checkpointLayer)
	}

	return batch
}

// processMicroNetBatchADHD processes micro networks with ADHD evaluation
func (n *Network[T]) processMicroNetBatchADHD(batch *CheckpointBatch, config *GrowConfig) []*MicroNetCandidate[T] {
	var candidates []*MicroNetCandidate[T]

	// Get baseline ADHD score
	baseMicroNet := n.ExtractMicroNetwork(batch.Layer)
	baseADHDScore := n.evaluateMicroNetADHD(baseMicroNet, batch)

	for i := 0; i < config.MicroNetCount; i++ {
		// Extract base micro network
		microNet := n.ExtractMicroNetwork(batch.Layer)

		// Try adding a layer
		improvedMicroNet := n.tryAddingLayerToMicroNet(microNet, config)

		if config.UseGPUForMicroNets {
			improvedMicroNet.Network.WebGPUNative = true
			if err := improvedMicroNet.Network.InitializeOptimizedGPU(); err != nil && config.Debug {
				fmt.Printf("⚠️  Failed to init GPU for micro net: %v\n", err)
			}
			defer improvedMicroNet.Network.CleanupOptimizedGPU()
		}

		// Train the improved micro network for 1 epoch as requested
		n.trainMicroNetOnBatchADHD(improvedMicroNet, batch, config)

		// Evaluate performance with ADHD
		adhdScore := n.evaluateMicroNetADHD(improvedMicroNet, batch)
		adhdImprovement := adhdScore - baseADHDScore

		candidate := &MicroNetCandidate[T]{
			MicroNet:        improvedMicroNet,
			ADHDScore:       adhdScore,
			ADHDImprovement: adhdImprovement,
			TrainingLoss:    0.0, // Not tracking loss, focusing on ADHD
		}

		candidates = append(candidates, candidate)
	}

	return candidates
}

// evaluateMicroNetADHD evaluates a micro network using ADHD metrics
func (n *Network[T]) evaluateMicroNetADHD(microNet *MicroNetwork[T], batch *CheckpointBatch) float64 {
	expectedOutputs := make([]float64, len(batch.States))
	actualOutputs := make([]float64, len(batch.States))

	for i := range batch.States {
		microNet.Network.ForwardFromLayer(microNet.CheckpointIdx, batch.States[i])
		output := microNet.Network.GetOutput()

		expectedOutputs[i] = float64(ArgMax(batch.Targets[i][0]))
		actualOutputs[i] = float64(ArgMax(output))
	}

	// Use ADHD evaluation on micro network
	microNet.Network.EvaluateModel(expectedOutputs, actualOutputs)
	return microNet.Network.ComputeFinalScore()
}

// trainMicroNetOnBatchADHD trains a micro network for exactly 1 epoch
func (n *Network[T]) trainMicroNetOnBatchADHD(microNet *MicroNetwork[T], batch *CheckpointBatch, config *GrowConfig) {
	if config.TrainingEpochs <= 0 {
		if config.Debug {
			fmt.Println("   ⏩ Skipping micro net training (TrainingEpochs = 0)")
		}
		return
	}

	var clipUpper, clipLower T
	var zero T

	// Set clipping bounds
	clipUpper = zero + T(1) // Always safe upper
	clipLower = zero        // Default lower for unsigned

	defer func() {
		if recover() != nil {
			clipLower = zero
		}
	}()

	// Attempt to detect signed type
	testVal := zero - T(1)
	_ = testVal
	clipLower = zero - T(1)

	trainedSamples := 0

	for epoch := 0; epoch < config.TrainingEpochs; epoch++ {
		for i := range batch.States {
			microNet.Network.ForwardFromLayer(microNet.CheckpointIdx, batch.States[i])
			microNet.Network.Backward(batch.Targets[i], config.LearningRate, clipUpper, clipLower)
			trainedSamples++
		}
	}

	if config.Debug && trainedSamples > 0 {
		fmt.Printf("   🎓 Trained micro net for %d epoch(s) on %d samples with lr=%.3f\n",
			config.TrainingEpochs, trainedSamples, config.LearningRate)
	}
}

// tryAddingLayerToMicroNet attempts to add a new layer to a micro network
func (n *Network[T]) tryAddingLayerToMicroNet(microNet *MicroNetwork[T], config *GrowConfig) *MicroNetwork[T] {
	// Create new network with additional layer
	originalLayers := microNet.Network.Layers

	newLayerSizes := []struct{ Width, Height int }{
		{originalLayers[0].Width, originalLayers[0].Height}, // Input
		{originalLayers[1].Width, originalLayers[1].Height}, // Checkpoint
		{config.NewLayerWidth, config.NewLayerHeight},       // New hidden layer
		{originalLayers[2].Width, originalLayers[2].Height}, // Output
	}

	newActivations := []string{
		originalLayers[0].Neurons[0][0].Activation,
		originalLayers[1].Neurons[0][0].Activation,
		config.NewLayerActivation,
		originalLayers[2].Neurons[0][0].Activation,
	}

	newFullyConnected := []bool{false, true, true, true}

	newNetwork := NewNetwork[T](newLayerSizes, newActivations, newFullyConnected)
	newNetwork.Debug = false

	// Copy existing weights
	microNet.Network.copyWeightsBetweenNetworks(0, 1, newNetwork, 0, 1) // Input → checkpoint
	microNet.adaptOutputWeights(newNetwork)                             // Adapt output weights

	return &MicroNetwork[T]{
		Network:       newNetwork,
		SourceLayers:  microNet.SourceLayers,
		CheckpointIdx: microNet.CheckpointIdx,
	}
}

// Helper methods for backwards compatibility
func (n *Network[T]) evaluateAccuracy(inputs, targets [][][]float64) float64 {
	// For backwards compatibility, but we prefer ADHD scoring
	return n.evaluateADHDScore(inputs, targets) / 100.0 // Convert ADHD score to 0-1 range
}

func (n *Network[T]) calculateSampleAccuracy(output, target []float64) float64 {
	predicted := ArgMax(output)
	expected := ArgMax(target)

	if predicted == expected {
		return 1.0
	}
	return 0.0
}

func (n *Network[T]) EvaluateFullScore(inputs, targets [][][]float64) float64 {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		n.Forward(inputs[i])
		out := n.GetOutput()
		expected[i] = float64(ArgMax(targets[i][0]))
		actual[i] = float64(ArgMax(out))
	}

	n.EvaluateFull(expected, actual)
	return n.Performance.Score
}
