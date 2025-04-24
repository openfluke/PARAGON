package paragon

//ADH - Accuracy Deviation Heatmap Distribution
/*
 Color Representation:
Red → High confidence, model is highly accurate (0-10% deviation).
Orange/Yellow → Medium error range (10%-50% deviation).
Blue → Predictions are significantly off (50%-100% deviation).
Black → Beyond 100% deviation (model is extremely wrong).
*/

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"
)

// ADHDBucket represents a specific deviation range
type ADHDBucket struct {
	RangeMin float64 `json:"range_min"`
	RangeMax float64 `json:"range_max"`
	Count    int     `json:"count"`
}

// ADHDResult represents the performance of the model on one evaluation
type ADHDResult struct {
	ExpectedOutput float64 `json:"expected"`
	ActualOutput   float64 `json:"actual"`
	Deviation      float64 `json:"deviation"`
	Bucket         string  `json:"bucket"`
}

// ADHDPerformance stores the full model performance breakdown
type ADHDPerformance struct {
	Buckets  map[string]ADHDBucket `json:"buckets"`
	Score    float64               `json:"score"`
	Total    int                   `json:"total_samples"`
	Failures int                   `json:"failures"`
}

// NewADHDPerformance initializes an empty performance struct
func NewADHDPerformance() *ADHDPerformance {
	return &ADHDPerformance{
		Buckets: map[string]ADHDBucket{
			"0-10%":   {0, 10, 0},
			"10-20%":  {10, 20, 0},
			"20-30%":  {20, 30, 0},
			"30-40%":  {30, 40, 0},
			"40-50%":  {40, 50, 0},
			"50-100%": {50, 100, 0},
			"100%+":   {100, math.Inf(1), 0}, // Inf represents anything beyond 100% deviation
		},
		Score:    0,
		Total:    0,
		Failures: 0,
	}
}

// EvaluatePrediction categorizes an expected vs actual output into an ADHD bucket
func (n *Network) EvaluatePrediction(expected, actual float64) ADHDResult {
	var deviation float64
	if math.Abs(expected) < 1e-10 { // Handle near-zero expected values
		deviation = math.Abs(actual-expected) * 100 // Scale to percentage
	} else {
		deviation = math.Abs((actual - expected) / expected * 100) // % error
	}

	var bucketName string
	switch {
	case deviation <= 10:
		bucketName = "0-10%"
	case deviation <= 20:
		bucketName = "10-20%"
	case deviation <= 30:
		bucketName = "20-30%"
	case deviation <= 40:
		bucketName = "30-40%"
	case deviation <= 50:
		bucketName = "40-50%"
	case deviation <= 100:
		bucketName = "50-100%"
	default:
		bucketName = "100%+"
	}

	return ADHDResult{
		ExpectedOutput: expected,
		ActualOutput:   actual,
		Deviation:      deviation,
		Bucket:         bucketName,
	}
}

// (n *Network) UpdateADHDPerformance updates the performance struct with a single result
func (n *Network) UpdateADHDPerformance(result ADHDResult) {
	bucket := n.Performance.Buckets[result.Bucket]
	bucket.Count++
	n.Performance.Buckets[result.Bucket] = bucket

	n.Performance.Total++
	if result.Bucket == "100%+" {
		n.Performance.Failures++
	}

	// Ensure deviation is never NaN or infinity
	if math.IsNaN(result.Deviation) || math.IsInf(result.Deviation, 0) {
		result.Deviation = 100 // Default worst case
	}

	// Compute score: lower deviations contribute more positively
	n.Performance.Score += math.Max(0, 100-result.Deviation)
}

func (n *Network) ComputeFinalScore() float64 {
	if n.Performance.Total == 0 || math.IsNaN(n.Performance.Score) || math.IsInf(n.Performance.Score, 0) {
		return 0 // Avoid NaN issues
	}
	return math.Max(0, n.Performance.Score/float64(n.Performance.Total)) // Ensure non-negative score
}

// (n *Network) EvaluateModel processes a batch of expected vs actual outputs
func (n *Network) EvaluateModel(expectedOutputs, actualOutputs []float64) {
	if len(expectedOutputs) != len(actualOutputs) {
		fmt.Println("Error: Mismatched expected vs actual data sizes.")
		return
	}

	n.Performance = NewADHDPerformance()

	for i := range expectedOutputs {
		result := n.EvaluatePrediction(expectedOutputs[i], actualOutputs[i])
		n.UpdateADHDPerformance(result)
	}

	n.Performance.Score = n.ComputeFinalScore()
}

// EvaluateFromCheckpoint evaluates ADHD metrics using checkpoint states
func (n *Network) EvaluateFromCheckpoint(checkpoints [][][]float64, expectedOutputs []float64, checkpointLayerIdx int) {
	if len(checkpoints) != len(expectedOutputs) {
		fmt.Printf("Error: Mismatched checkpoints (%d) vs expected outputs (%d) sizes.\n", len(checkpoints), len(expectedOutputs))
		return
	}

	n.Performance = NewADHDPerformance()
	actualOutputs := make([]float64, len(expectedOutputs))

	for i := range checkpoints {
		// Compute output from checkpoint
		n.ForwardFromLayer(checkpointLayerIdx, checkpoints[i])
		outputLayer := n.Layers[n.OutputLayer]
		outputValues := make([]float64, outputLayer.Width)
		for x := 0; x < outputLayer.Width; x++ {
			outputValues[x] = outputLayer.Neurons[0][x].Value
		}
		pred := ArgMax(outputValues)
		actualOutputs[i] = float64(pred)
	}

	for i := range expectedOutputs {
		result := n.EvaluatePrediction(expectedOutputs[i], actualOutputs[i])
		n.UpdateADHDPerformance(result)
	}

	n.Performance.Score = n.ComputeFinalScore()
}

// EvaluateFromCheckpointFilesWithTiming loads checkpoint files, runs the forward pass
// from the given checkpoint layer, and evaluates the ADHD score.
// It returns the computed score along with the total file load and forward times.
func (n *Network) EvaluateFromCheckpointFilesWithTiming(checkpointFiles []string, expectedOutputs []float64, checkpointLayerIdx int) (score float64, totalLoadTime, totalForwardTime time.Duration) {
	// Validate that the number of checkpoint files matches the expected outputs.
	if len(checkpointFiles) != len(expectedOutputs) {
		fmt.Printf("Error: Mismatched checkpoint files (%d) vs expected outputs (%d) sizes.\n", len(checkpointFiles), len(expectedOutputs))
		return 0, 0, 0
	}

	// Reset performance tracking.
	n.Performance = NewADHDPerformance()
	actualOutputs := make([]float64, len(checkpointFiles))

	// Process each checkpoint file.
	for i, cpFile := range checkpointFiles {
		startLoad := time.Now()
		data, err := os.ReadFile(cpFile)
		if err != nil {
			log.Printf("Failed to read checkpoint file %s: %v", cpFile, err)
			continue // Skip sample on error
		}
		var cpState [][]float64
		if err := json.Unmarshal(data, &cpState); err != nil {
			log.Printf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
			continue // Skip sample on error
		}
		totalLoadTime += time.Since(startLoad)

		startForward := time.Now()
		n.ForwardFromLayer(checkpointLayerIdx, cpState)
		totalForwardTime += time.Since(startForward)

		out := n.ExtractOutput()
		pred := ArgMax(out)
		actualOutputs[i] = float64(pred)
	}

	// Evaluate each prediction.
	for i := range expectedOutputs {
		result := n.EvaluatePrediction(expectedOutputs[i], actualOutputs[i])
		n.UpdateADHDPerformance(result)
	}
	n.Performance.Score = n.ComputeFinalScore()
	return n.Performance.Score, totalLoadTime, totalForwardTime
}

// ExtractOutput returns the output layer values as a slice.
func (n *Network) ExtractOutput() []float64 {
	outWidth := n.Layers[n.OutputLayer].Width
	output := make([]float64, outWidth)
	for x := 0; x < outWidth; x++ {
		output[x] = n.Layers[n.OutputLayer].Neurons[0][x].Value
	}
	return output
}
