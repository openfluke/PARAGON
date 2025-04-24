// data_utils.go
package paragon

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
)

// SplitDataset takes inputs/targets and splits them into train/validation sets
// according to the given fraction (e.g., 0.8 for 80% train, 20% val).
func SplitDataset(inputs [][][]float64, targets [][][]float64, trainFrac float64) (
	trainIn [][][]float64, trainTarg [][][]float64,
	valIn [][][]float64, valTarg [][][]float64,
) {
	if len(inputs) != len(targets) {
		panic("inputs and targets length mismatch!")
	}
	n := len(inputs)
	trainSize := int(trainFrac * float64(n))

	perm := rand.Perm(n)
	trainIn = make([][][]float64, trainSize)
	trainTarg = make([][][]float64, trainSize)
	valIn = make([][][]float64, n-trainSize)
	valTarg = make([][][]float64, n-trainSize)

	for i, p := range perm {
		if i < trainSize {
			trainIn[i] = inputs[p]
			trainTarg[i] = targets[p]
		} else {
			valIn[i-trainSize] = inputs[p]
			valTarg[i-trainSize] = targets[p]
		}
	}
	return
}

// Cleaner processes the given data array by cleaning specified columns.
// For columns in nameCols, automatically detects and removes the common prefix from cube names (e.g., "[ARC]-OC-gen1-v0-POD_192.168.0.227_10023_head" to "head").
// For columns in paramCols, extracts the value from key:value pairs (e.g., "motor_target_velocity:0" to "0").
// Returns a new [][]string array with the cleaned data.
func Cleaner(data [][]string, nameCols, paramCols []int) ([][]string, error) {
	if len(data) == 0 {
		return [][]string{}, nil
	}

	// Create a new array to store the cleaned data
	cleanedData := make([][]string, len(data))
	for i := range cleanedData {
		cleanedData[i] = make([]string, len(data[i]))
		copy(cleanedData[i], data[i])
	}

	// Determine the prefix for each name column
	prefixes := make(map[int]string)
	for _, col := range nameCols {
		if col < 0 || col >= len(data[0]) {
			return nil, fmt.Errorf("invalid name column index %d for row with %d columns", col, len(data[0]))
		}
		// Use the first row to determine the prefix
		if len(data) == 0 {
			continue
		}
		name := data[0][col]
		// Find the last underscore to determine the prefix
		lastUnderscore := strings.LastIndex(name, "_")
		if lastUnderscore == -1 {
			return nil, fmt.Errorf("no underscore found in name %s in column %d; cannot determine prefix", name, col)
		}
		prefix := name[:lastUnderscore+1] // Include the underscore in the prefix
		prefixes[col] = prefix
	}

	// Process each row
	for rowIdx, row := range cleanedData {
		// Clean cube names in specified columns
		for _, col := range nameCols {
			prefix, exists := prefixes[col]
			if !exists {
				continue // Skip if no prefix was determined (e.g., empty data)
			}
			// Remove the prefix from the cube name
			cleanedData[rowIdx][col] = strings.TrimPrefix(row[col], prefix)
		}

		// Clean joint parameters in specified columns
		for _, col := range paramCols {
			if col < 0 || col >= len(row) {
				return nil, fmt.Errorf("invalid param column index %d for row with %d columns", col, len(row))
			}
			// Split the parameter on ":" and take the value part
			parts := strings.Split(row[col], ":")
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid joint parameter format in row %d, column %d: %s", rowIdx, col, row[col])
			}
			cleanedData[rowIdx][col] = parts[1]
		}
	}

	return cleanedData, nil
}

// PrintTable prints the given data array in a table-like format with aligned columns.
func PrintTable(data [][]string) {
	if len(data) == 0 {
		fmt.Println("No data to display.")
		return
	}

	// Determine the number of columns
	numCols := len(data[0])

	// Calculate the maximum width for each column
	colWidths := make([]int, numCols)
	for _, row := range data {
		if len(row) != numCols {
			fmt.Println("Inconsistent number of columns in data; cannot print table.")
			return
		}
		for j, cell := range row {
			if len(cell) > colWidths[j] {
				colWidths[j] = len(cell)
			}
		}
	}

	// Calculate the total width of the table (including padding and borders)
	totalWidth := 0
	for _, width := range colWidths {
		totalWidth += width + 3 // 2 spaces padding + 1 for the border
	}
	totalWidth += 1 // Final border

	// Print the top border
	fmt.Println(strings.Repeat("-", totalWidth))

	// Print each row
	for _, row := range data {
		fmt.Print("|")
		for j, cell := range row {
			// Pad the cell to the column width
			padding := colWidths[j] - len(cell)
			fmt.Printf(" %s%s |", cell, strings.Repeat(" ", padding))
		}
		fmt.Println()
	}

	// Print the bottom border
	fmt.Println(strings.Repeat("-", totalWidth))
}

// Converter maps unique strings in specified columns to integers and replaces them in the data.
// Returns the mapping as [][2]string (e.g., [["head", "0"], ["body", "1"]]) and the new data with integers.
func Converter(data [][]string, colsToConvert []int) ([][]string, [][]string, error) {
	if len(data) == 0 {
		return [][]string{}, [][]string{}, nil
	}

	// Validate column indices
	numCols := len(data[0])
	for _, col := range colsToConvert {
		if col < 0 || col >= numCols {
			return nil, nil, fmt.Errorf("invalid column index %d for data with %d columns", col, numCols)
		}
	}

	// Collect unique strings from specified columns
	uniqueStrings := make(map[string]bool)
	for _, row := range data {
		if len(row) != numCols {
			return nil, nil, fmt.Errorf("inconsistent number of columns in row: expected %d, got %d", numCols, len(row))
		}
		for _, col := range colsToConvert {
			uniqueStrings[row[col]] = true
		}
	}

	// Create a mapping of unique strings to integers
	stringToInt := make(map[string]int)
	intToString := make(map[int]string)
	uniqueList := make([]string, 0, len(uniqueStrings))
	for s := range uniqueStrings {
		uniqueList = append(uniqueList, s)
	}
	// Sort the list for consistent ordering (optional, but ensures reproducibility)
	for i, s := range uniqueList {
		stringToInt[s] = i
		intToString[i] = s
	}

	// Create the mapping array
	mapping := make([][]string, len(uniqueList))
	for i, s := range uniqueList {
		mapping[i] = []string{s, strconv.Itoa(i)}
	}

	// Create a new data array with replaced values
	newData := make([][]string, len(data))
	for i := range newData {
		newData[i] = make([]string, len(data[i]))
		copy(newData[i], data[i])
	}

	// Replace strings with their integer mappings in specified columns
	for rowIdx, row := range newData {
		for _, col := range colsToConvert {
			value := row[col]
			intValue := stringToInt[value]
			newData[rowIdx][col] = strconv.Itoa(intValue)
		}
	}

	return mapping, newData, nil
}

// ConvertToFloat64 converts [][]string into [][]float64.
// It returns an error if any value cannot be parsed.
func ConvertToFloat64(data [][]string) ([][]float64, error) {
	converted := make([][]float64, len(data))
	for i, row := range data {
		converted[i] = make([]float64, len(row))
		for j, cell := range row {
			val, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to convert data[%d][%d] = %q to float64: %v", i, j, cell, err)
			}
			converted[i][j] = val
		}
	}
	return converted, nil
}

// Padding pads each row in the data to have exactly targetCols columns by appending paddingValue.
// Rows with more columns than targetCols are truncated to targetCols.
func Padding(data [][]float64, targetCols int, paddingValue float64) [][]float64 {
	paddedData := make([][]float64, len(data))
	for i, row := range data {
		currentCols := len(row)
		if currentCols == targetCols {
			paddedData[i] = row
		} else if currentCols < targetCols {
			// Pad with paddingValue if fewer columns
			padding := make([]float64, targetCols-currentCols)
			for j := range padding {
				padding[j] = paddingValue
			}
			paddedData[i] = append(row, padding...)
		} else {
			// Truncate if more columns
			paddedData[i] = row[:targetCols]
		}
	}
	return paddedData
}

// RemoveColumns removes the specified columns from each row of the data.
func RemoveColumns(data [][]float64, columnsToRemove []int) [][]float64 {
	// Create a map for quick lookup of columns to remove
	removeMap := make(map[int]bool)
	for _, col := range columnsToRemove {
		removeMap[col] = true
	}

	// Initialize the result slice with the same number of rows as the input
	result := make([][]float64, len(data))

	// Process each row
	for i, row := range data {
		newRow := []float64{}
		// Iterate over each column in the row
		for j, val := range row {
			// Include the value if its index is not in the remove map
			if !removeMap[j] {
				newRow = append(newRow, val)
			}
		}
		result[i] = newRow
	}

	return result
}

// SelectColumns creates a new 2D slice containing only the specified columns from each row.
// If a column index is out of range or negative, it uses "0.0" as the default value.
// The order of columns in the output matches the order in columnsToKeep.
func SelectColumns(data [][]float64, columnsToKeep []int) [][]float64 {
	result := make([][]float64, len(data))
	for i, row := range data {
		newRow := make([]float64, len(columnsToKeep))
		for j, col := range columnsToKeep {
			if col >= 0 && col < len(row) {
				newRow[j] = row[col]
			} else {
				newRow[j] = 0.0
			}
		}
		result[i] = newRow
	}
	return result
}
