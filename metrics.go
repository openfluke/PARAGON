// metrics.go
package paragon

import (
	"fmt"
	"math"
	"sort"
)

// CompositePerformance holds detailed diagnostic info for scalar or vector output models.
type CompositePerformance struct {
	ADHD            *ADHDPerformance
	ExactMatchCount int
	TotalSamples    int
	MeanAbsError    float64
	MeanPctError    float64
	StdAbsError     float64
	WorstErrors     []ErrorSample
	Score           float64 // Unified score
}

// ErrorSample holds per-sample deviation info
type ErrorSample struct {
	Index    int
	Expected float64
	Actual   float64
	AbsError float64
	PctError float64
}

type SamplePerformance struct {
	ExactMatchCount int
	TotalSamples    int
	MeanAbsError    float64
	MeanPctError    float64
	StdAbsError     float64
	ADHDScore       float64
	CompositeScore  float64
	WorstSamples    []struct {
		Index    int
		AbsMean  float64
		PctError float64
	}
	DeviationBuckets map[string]ADHDBucket
}

// ComputeAccuracy calculates the classification accuracy for a dataset.
func ComputeAccuracy(nn *Network, inputs [][][]float64, targets [][][]float64) float64 {
	if len(inputs) == 0 {
		return 0
	}
	correct := 0
	for i := range inputs {
		nn.Forward(inputs[i])

		// Suppose the output layer is 1D or 2D with (Height=1), and we have a one-hot target.
		// This is typical for classification, but can be extended as needed.
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}

		pred := argMax(outputValues)
		label := argMax(targets[i][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// EvaluateWithADHD runs the ADHD evaluation on a classification dataset.
func EvaluateWithADHD(nn *Network, inputs [][][]float64, targets [][][]float64) {
	if len(inputs) == 0 {
		return
	}

	expectedOutputs := make([]float64, len(inputs))
	actualOutputs := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}
		pred := argMax(outputValues)
		label := argMax(targets[i][0])

		expectedOutputs[i] = float64(label)
		actualOutputs[i] = float64(pred)
	}

	nn.EvaluateModel(expectedOutputs, actualOutputs) // The ADHD logic from adhd.go
}

// argMax finds the index of the largest value in a 1D slice.
func argMax(arr []float64) int {
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func (n *Network) EvaluateFull(expected, actual []float64) {
	n.Performance = NewADHDPerformance()

	errors := []ErrorSample{}
	var totalAbs, totalPct, sumSq float64
	exact := 0

	for i := range expected {
		exp := expected[i]
		act := actual[i]
		abs := math.Abs(exp - act)
		pct := 0.0
		if math.Abs(exp) > 1e-6 {
			pct = (abs / math.Abs(exp)) * 100
		}

		if abs < 1e-6 {
			exact++
		}

		totalAbs += abs
		totalPct += pct
		sumSq += abs * abs

		errors = append(errors, ErrorSample{
			Index:    i,
			Expected: exp,
			Actual:   act,
			AbsError: abs,
			PctError: pct,
		})

		res := n.EvaluatePrediction(exp, act)
		n.UpdateADHDPerformance(res)
	}

	// ✅ Fix: Normalize ADHD score after accumulation
	n.Performance.Score = n.ComputeFinalScore()

	// Sort for worst errors
	sort.Slice(errors, func(i, j int) bool {
		return errors[i].PctError > errors[j].PctError
	})

	n.Composite = &CompositePerformance{
		ADHD:            n.Performance,
		ExactMatchCount: exact,
		TotalSamples:    len(expected),
		MeanAbsError:    totalAbs / float64(len(expected)),
		MeanPctError:    totalPct / float64(len(expected)),
		StdAbsError:     math.Sqrt(sumSq / float64(len(expected))),
		WorstErrors:     errors[:min(5, len(errors))],

		// Composite score is a simple blend of ADHD score and Exact Match rate
		Score: (n.Performance.Score + float64(exact)*100/float64(len(expected))) / 2,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (n *Network) PrintFullDiagnostics() {
	if n.Composite == nil {
		fmt.Println("⚠️ No diagnostics found. Run EvaluateFull() first.")
		return
	}
	p := n.Composite

	fmt.Println("🧠 Full Composite Performance Report")
	fmt.Println("===================================")
	fmt.Printf("📦 Samples Evaluated: %d\n", p.TotalSamples)
	fmt.Printf("✅ Exact Matches: %d (%.2f%%)\n", p.ExactMatchCount, float64(p.ExactMatchCount)*100/float64(p.TotalSamples))
	fmt.Printf("📉 Mean Absolute Error: %.4f\n", p.MeanAbsError)
	fmt.Printf("📐 Mean %% Deviation: %.2f%%\n", p.MeanPctError)
	fmt.Printf("📊 Std Dev of Abs Error: %.4f\n", p.StdAbsError)
	fmt.Printf("🧮 ADHD Score: %.2f\n", p.ADHD.Score)
	fmt.Printf("🧮 Composite Score: %.2f\n", p.Score)
	fmt.Println("📊 Deviation Buckets:")
	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, k := range keys {
		b := p.ADHD.Buckets[k]
		fmt.Printf(" - %s → %d samples\n", k, b.Count)
	}
	fmt.Println("🚨 Worst 5 Samples:")
	for _, e := range p.WorstErrors {
		fmt.Printf("   [%d] Expected=%.3f, Actual=%.3f | Abs=%.3f | %%=%.2f%%\n",
			e.Index, e.Expected, e.Actual, e.AbsError, e.PctError)
	}
}

func ComputePerSamplePerformance(expectedVectors, actualVectors [][]float64, epsilon float64, net *Network) *SamplePerformance {
	net.Performance = NewADHDPerformance()
	bucketMap := make(map[string]ADHDBucket)

	type SampleResult struct {
		Index    int
		AbsMean  float64
		PctError float64
	}
	results := []SampleResult{}

	var totalAbs, totalPct, sumSq float64
	exact := 0
	sampleCount := len(expectedVectors)

	for i := range expectedVectors {
		exp := expectedVectors[i]
		act := actualVectors[i]
		if len(exp) != len(act) {
			fmt.Printf("⚠️ Sample %d size mismatch: exp=%d, act=%d\n", i, len(exp), len(act))
			continue
		}

		vecLen := len(exp)
		var sumAbs, sumPct float64
		match := true

		for j := 0; j < vecLen; j++ {
			abs := math.Abs(exp[j] - act[j])
			pct := 0.0
			if math.Abs(exp[j]) > 1e-6 {
				pct = (abs / math.Abs(exp[j])) * 100
			}
			sumAbs += abs
			sumPct += pct
			if abs > epsilon {
				match = false
			}
		}

		meanAbs := sumAbs / float64(vecLen)
		meanPct := sumPct / float64(vecLen)
		totalAbs += meanAbs
		totalPct += meanPct
		sumSq += meanAbs * meanAbs

		if match {
			exact++
		}

		// Simulate a soft scalar deviation evaluation (for ADHD usage)
		result := net.EvaluatePrediction(1.0, 1.0-meanPct/100.0)
		net.UpdateADHDPerformance(result)

		// Count the sample into the appropriate bucket
		b := bucketMap[result.Bucket]
		b.Count++
		bucketMap[result.Bucket] = b

		results = append(results, SampleResult{i, meanAbs, meanPct})
	}

	net.Performance.Score = net.ComputeFinalScore()

	sort.Slice(results, func(i, j int) bool {
		return results[i].PctError > results[j].PctError
	})

	// Extract top 5 worst
	worst := make([]struct {
		Index    int
		AbsMean  float64
		PctError float64
	}, min(5, len(results)))
	for i := 0; i < len(worst); i++ {
		worst[i] = struct {
			Index    int
			AbsMean  float64
			PctError float64
		}{
			Index:    results[i].Index,
			AbsMean:  results[i].AbsMean,
			PctError: results[i].PctError,
		}
	}

	exactPct := 100 * float64(exact) / float64(sampleCount)
	composite := (net.Performance.Score + exactPct) / 2

	return &SamplePerformance{
		ExactMatchCount:  exact,
		TotalSamples:     sampleCount,
		MeanAbsError:     totalAbs / float64(sampleCount),
		MeanPctError:     totalPct / float64(sampleCount),
		StdAbsError:      math.Sqrt(sumSq / float64(sampleCount)),
		ADHDScore:        net.Performance.Score,
		CompositeScore:   composite,
		WorstSamples:     worst,
		DeviationBuckets: bucketMap,
	}
}

func PrintSampleDiagnostics(p *SamplePerformance, epsilon float64) {
	fmt.Println("🧠 Sample-Level Evaluation (per vector)")
	fmt.Println("======================================")
	fmt.Printf("🧪 Total Samples: %d\n", p.TotalSamples)
	fmt.Printf("✅ Exact Matches (ε=%.4f): %d (%.2f%%)\n", epsilon, p.ExactMatchCount, 100*float64(p.ExactMatchCount)/float64(p.TotalSamples))
	fmt.Printf("📉 Mean Absolute Error (per sample): %.4f\n", p.MeanAbsError)
	fmt.Printf("📐 Mean %% Deviation (per sample): %.2f%%\n", p.MeanPctError)
	fmt.Printf("📊 Std Dev of Abs Error: %.4f\n", p.StdAbsError)
	fmt.Printf("🧮 ADHD Score (sample-level view): %.2f\n", p.ADHDScore)
	fmt.Printf("🧮 Composite Score (ADHD + Exact): %.2f\n", p.CompositeScore)

	fmt.Println("📊 Deviation Buckets:")
	knownBuckets := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, k := range knownBuckets {
		if b, ok := p.DeviationBuckets[k]; ok {
			fmt.Printf(" - %s → %d samples\n", k, b.Count)
		} else {
			fmt.Printf(" - %s → 0 samples\n", k)
		}
	}

	fmt.Println("🚨 Worst 5 Samples (by % deviation):")
	for _, r := range p.WorstSamples {
		fmt.Printf("   [%d] MAE=%.4f | %%=%.2f%%\n", r.Index, r.AbsMean, r.PctError)
	}
}
