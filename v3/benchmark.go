package paragon

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"
)

type BenchmarkResult struct {
	Type   string `json:"type"`
	Single int    `json:"single_threaded_ops"`
	Multi  int    `json:"multi_threaded_ops"`
}

// BenchmarkNumericOps benchmarks multiply-add ops for any Numeric type `T`.
func BenchmarkNumericOps[T Numeric](label string, duration time.Duration, multiThreaded bool) int {
	if multiThreaded {
		return runMultiThreadedOps[T](duration)
	}
	return runSingleThreadedOps[T](duration)
}

// runSingleThreadedOps performs multiply-adds on a single thread.
func runSingleThreadedOps[T Numeric](duration time.Duration) int {
	var a, b T = T(1), T(2)
	ops := 0
	start := time.Now()
	for time.Since(start) < duration {
		a *= b
		b += a
		ops++
	}
	return ops
}

// runMultiThreadedOps benchmarks ops across all CPU cores.
func runMultiThreadedOps[T Numeric](duration time.Duration) int {
	var wg sync.WaitGroup
	numCores := runtime.NumCPU()
	opsChan := make(chan int, numCores)

	for i := 0; i < numCores; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var a, b T = T(1), T(2)
			ops := 0
			start := time.Now()
			for time.Since(start) < duration {
				a *= b
				b += a
				ops++
			}
			opsChan <- ops
		}()
	}

	wg.Wait()
	close(opsChan)

	totalOps := 0
	for val := range opsChan {
		totalOps += val
	}
	return totalOps
}

// RunAllBenchmarks runs benchmarks across all Numeric types.
func RunAllBenchmarks(duration time.Duration) string {
	var results []BenchmarkResult

	types := []struct {
		name string
		run  func() (int, int)
	}{
		{"int", func() (int, int) {
			return BenchmarkNumericOps[int]("int", duration, false), BenchmarkNumericOps[int]("int", duration, true)
		}},
		{"int8", func() (int, int) {
			return BenchmarkNumericOps[int8]("int8", duration, false), BenchmarkNumericOps[int8]("int8", duration, true)
		}},
		{"int16", func() (int, int) {
			return BenchmarkNumericOps[int16]("int16", duration, false), BenchmarkNumericOps[int16]("int16", duration, true)
		}},
		{"int32", func() (int, int) {
			return BenchmarkNumericOps[int32]("int32", duration, false), BenchmarkNumericOps[int32]("int32", duration, true)
		}},
		{"int64", func() (int, int) {
			return BenchmarkNumericOps[int64]("int64", duration, false), BenchmarkNumericOps[int64]("int64", duration, true)
		}},
		{"uint", func() (int, int) {
			return BenchmarkNumericOps[uint]("uint", duration, false), BenchmarkNumericOps[uint]("uint", duration, true)
		}},
		{"uint8", func() (int, int) {
			return BenchmarkNumericOps[uint8]("uint8", duration, false), BenchmarkNumericOps[uint8]("uint8", duration, true)
		}},
		{"uint16", func() (int, int) {
			return BenchmarkNumericOps[uint16]("uint16", duration, false), BenchmarkNumericOps[uint16]("uint16", duration, true)
		}},
		{"uint32", func() (int, int) {
			return BenchmarkNumericOps[uint32]("uint32", duration, false), BenchmarkNumericOps[uint32]("uint32", duration, true)
		}},
		{"uint64", func() (int, int) {
			return BenchmarkNumericOps[uint64]("uint64", duration, false), BenchmarkNumericOps[uint64]("uint64", duration, true)
		}},
		{"float32", func() (int, int) {
			return BenchmarkNumericOps[float32]("float32", duration, false), BenchmarkNumericOps[float32]("float32", duration, true)
		}},
		{"float64", func() (int, int) {
			return BenchmarkNumericOps[float64]("float64", duration, false), BenchmarkNumericOps[float64]("float64", duration, true)
		}},
	}

	// Run and collect results with delay
	for _, t := range types {
		single, multi := t.run()
		results = append(results, BenchmarkResult{
			Type:   t.name,
			Single: single,
			Multi:  multi,
		})
		time.Sleep(500 * time.Millisecond)
	}

	// Pretty print
	fmt.Printf("Benchmark Results (Duration: %v)\n", duration)
	fmt.Println("----------------------------------------------------------")
	fmt.Printf("%-10s | %-15s | %-15s\n", "Type", "Single-Threaded", "Multi-Threaded")
	fmt.Println("----------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-10s | %-15s | %-15s\n", r.Type, formatNumber(r.Single), formatNumber(r.Multi))
	}
	fmt.Println("----------------------------------------------------------")

	// Convert to JSON
	jsonData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Sprintf(`{"error": "failed to serialize benchmark results: %v"}`, err)
	}
	return string(jsonData)
}

// formatNumber formats large integers with readable suffixes.
func formatNumber(num int) string {
	switch {
	case float64(num) >= 1e12:
		return fmt.Sprintf("%.2fT", float64(num)/1e12)
	case float64(num) >= 1e9:
		return fmt.Sprintf("%.2fB", float64(num)/1e9)
	case float64(num) >= 1e6:
		return fmt.Sprintf("%.2fM", float64(num)/1e6)
	case float64(num) >= 1e3:
		return fmt.Sprintf("%.2fK", float64(num)/1e3)
	default:
		return fmt.Sprintf("%d", num)
	}
}
