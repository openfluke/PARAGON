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
	"fmt"
	"math"
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

// (n *Network) EvaluatePrediction categorizes an expected vs actual output into an ADHD bucket
func (n *Network) EvaluatePrediction(expected, actual float64) ADHDResult {
	deviation := math.Abs((actual - expected) / expected * 100) // % error
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
