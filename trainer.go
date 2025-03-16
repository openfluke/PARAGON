package paragon

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TrainConfig defines parameters for a standard training loop that can be
// reused for various datasets and tasks.
type TrainConfig struct {
	Epochs           int     // Maximum number of epochs
	LearningRate     float64 // Base learning rate
	PlateauThreshold float64 // If change in loss is below this threshold...
	PlateauLimit     int     // ... for this many checks, we consider it a plateau
	EarlyStopAcc     float64 // If validation accuracy >= this, stop early (0 means disabled)
	Debug            bool    // If true, print debug logs during training
}

// Trainer is a reusable component that orchestrates the training process.
type Trainer struct {
	Network     *Network
	Config      TrainConfig
	HasAddedNeu bool // track if we’ve already added neurons (for demonstration)
}

// TrainWithValidation runs a training loop with an optional validation set.
// You can incorporate your custom logic for plateau detection or auto-layer expansion.
func (t *Trainer) TrainWithValidation(
	trainInputs [][][]float64, trainTargets [][][]float64,
	valInputs [][][]float64, valTargets [][][]float64,
	testInputs [][][]float64, testTargets [][][]float64,
) {

	epochsPerPhase := 50 // how many epochs to do per “phase”
	plateauCount := 0
	plateauLimit := t.Config.PlateauLimit
	plateauThreshold := t.Config.PlateauThreshold
	lr := t.Config.LearningRate
	prevLoss := math.Inf(1)
	totalEpochs := 0

	for totalEpochs < t.Config.Epochs {
		// Phase loop
		for epoch := 0; epoch < epochsPerPhase && totalEpochs < t.Config.Epochs; epoch++ {
			totalLoss := 0.0

			// Shuffle data each epoch
			perm := rand.Perm(len(trainInputs))
			shuffledInputs := make([][][]float64, len(trainInputs))
			shuffledTargets := make([][][]float64, len(trainTargets))
			for i, p := range perm {
				shuffledInputs[i] = trainInputs[p]
				shuffledTargets[i] = trainTargets[p]
			}

			// Mini-batch or full-batch loop
			for i := 0; i < len(shuffledInputs); i++ {
				t.Network.Forward(shuffledInputs[i])
				loss := t.Network.ComputeLoss(shuffledTargets[i])
				if math.IsNaN(loss) {
					if t.Config.Debug {
						fmt.Printf("NaN loss detected at sample %d, epoch %d\n", i, totalEpochs)
					}
					continue
				}
				totalLoss += loss
				t.Network.Backward(shuffledTargets[i], lr)
			}
			avgLoss := totalLoss / float64(len(trainInputs))
			if t.Config.Debug {
				fmt.Printf("Epoch %d, Training Loss: %.4f\n", totalEpochs, avgLoss)
			}

			// Plateau check
			lossChange := math.Abs(prevLoss - avgLoss)
			if lossChange < plateauThreshold {
				plateauCount++
				if t.Config.Debug {
					fmt.Printf("Plateau detected (%d/%d), loss change: %.6f\n",
						plateauCount, plateauLimit, lossChange)
				}
			} else {
				plateauCount = 0
			}
			prevLoss = avgLoss

			// If plateau repeated N times, demonstrate logic of adding neurons/layers
			if plateauCount >= plateauLimit {
				if !t.HasAddedNeu {
					fmt.Println("Loss plateaued, adding 20 neurons to layer 1 (demo).")
					t.Network.AddNeuronsToLayer(1, 20)
					t.HasAddedNeu = true
					plateauCount = 0
				} else {
					fmt.Println("Loss plateaued again, adding a new layer (demo).")
					t.Network.AddLayer(2, 8, 8, "leaky_relu", true)
					// (In a real scenario, you might do more sophisticated logic here.)
					plateauCount = 0
					break
				}
			}

			// Evaluate on training, validation, and test sets
			trainAcc := ComputeAccuracy(t.Network, trainInputs, trainTargets)
			valAcc := ComputeAccuracy(t.Network, valInputs, valTargets)
			testAcc := ComputeAccuracy(t.Network, testInputs, testTargets)

			fmt.Printf("After %d epochs:\n", totalEpochs+1)
			fmt.Printf("  Training Accuracy: %.2f%%\n", trainAcc*100)
			fmt.Printf("  Validation Accuracy: %.2f%%\n", valAcc*100)
			fmt.Printf("  Test Accuracy: %.2f%%\n", testAcc*100)

			// Early stopping condition
			if t.Config.EarlyStopAcc > 0 && valAcc >= t.Config.EarlyStopAcc {
				fmt.Println("Early stop triggered (Validation accuracy).")
				return
			}
			totalEpochs++
		}
	}
}

// TrainSimple runs a simpler training loop without plateau or fancy checks.
// Good for quick tasks.
func (t *Trainer) TrainSimple(
	trainInputs [][][]float64, trainTargets [][][]float64,
	epochs int,
) {
	for e := 0; e < epochs; e++ {
		start := time.Now()
		totalLoss := 0.0
		perm := rand.Perm(len(trainInputs))
		for i := 0; i < len(perm); i++ {
			idx := perm[i]
			t.Network.Forward(trainInputs[idx])
			loss := t.Network.ComputeLoss(trainTargets[idx])
			if !math.IsNaN(loss) {
				totalLoss += loss
				t.Network.Backward(trainTargets[idx], t.Config.LearningRate)
			}
		}
		fmt.Printf("Epoch %d, Loss: %.4f, Time: %v\n", e, totalLoss/float64(len(trainInputs)), time.Since(start))
	}
}
