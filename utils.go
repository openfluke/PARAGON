package paragon

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
)

// Softmax computes the softmax of a slice
func Softmax(inputs []float64) []float64 {
	maxVal := inputs[0]
	for _, v := range inputs {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expInputs := make([]float64, len(inputs))
	for i, v := range inputs {
		expInputs[i] = math.Exp(v - maxVal)
		expSum += expInputs[i]
	}
	for i := range expInputs {
		expInputs[i] /= expSum
	}
	return expInputs
}

// ArgMax returns the index of the maximum value in the slice.
// If the slice is empty, it returns -1.
func ArgMax(arr []float64) int {
	if len(arr) == 0 {
		return -1
	}
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

// FileExists checks if a file exists at the given path and returns true if it does, false otherwise.
func FileExists(filename string) bool {
	_, err := os.Stat(filename)
	if err == nil {
		return true // File exists
	}
	if os.IsNotExist(err) {
		return false // File does not exist
	}
	// If there's another error (e.g., permission denied), assume the file doesn't exist for safety
	return false
}

// ReadCSV reads a CSV file without headers and returns its contents as a [][]string array.
// Each row in the CSV file becomes a slice of strings in the returned array.
func ReadCSV(filename string) ([][]string, error) {
	// Check if the file exists
	if !FileExists(filename) {
		return nil, fmt.Errorf("file %s does not exist", filename)
	}

	// Open the CSV file
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file %s: %v", filename, err)
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	// Read all rows from the CSV file
	data, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV file %s: %v", filename, err)
	}

	// If the file is empty, return an empty slice
	if len(data) == 0 {
		return [][]string{}, nil
	}

	return data, nil
}
