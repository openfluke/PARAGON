package paragon

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

type GPUProcess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

func (n *Network[T]) StartGPUProcess() error {
	cmd := exec.Command("./gpu_forward_layer")
	stdin, _ := cmd.StdinPipe()
	rawStdout, _ := cmd.StdoutPipe()
	stdout := bufio.NewReader(rawStdout)
	if err := cmd.Start(); err != nil {
		return err
	}
	n.gpuProc = &GPUProcess{cmd, stdin, stdout}
	return nil
}

func (n *Network[T]) StopGPUProcess() error {
	if n.gpuProc != nil {
		n.gpuProc.stdin.Close()
		return n.gpuProc.cmd.Wait()
	}
	return nil
}

func (n *Network[T]) forwardLayerWebGPU(l int) bool {
	curr := n.Layers[l]

	// 1) Build payload
	type neuronJSON struct {
		Bias    float64   `json:"bias"`
		Inputs  []float64 `json:"inputs"`
		Weights []float64 `json:"weights"`
		Act     string    `json:"activation"`
	}
	layerData := make([]neuronJSON, 0, curr.Width*curr.Height)
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			cell := curr.Neurons[y][x]
			inp := make([]float64, len(cell.Inputs))
			wts := make([]float64, len(cell.Inputs))
			for i, c := range cell.Inputs {
				inp[i] = float64(any(n.Layers[c.SourceLayer].
					Neurons[c.SourceY][c.SourceX].Value).(T))
				wts[i] = float64(any(c.Weight).(T))
			}
			layerData = append(layerData, neuronJSON{
				Bias:    float64(any(cell.Bias).(T)),
				Inputs:  inp,
				Weights: wts,
				Act:     cell.Activation,
			})
		}
	}

	payload, err := json.Marshal(layerData)
	if err != nil {
		fmt.Println("❌ JSON marshal failed:", err)
		return false
	}

	// 2) Send payload to persistent GPU process
	_, err = n.gpuProc.stdin.Write(payload)
	if err != nil {
		fmt.Println("❌ Failed to write to GPU stdin:", err)
		return false
	}
	_, err = n.gpuProc.stdin.Write([]byte("\n"))
	if err != nil {
		fmt.Println("❌ Failed to write newline to GPU stdin:", err)
		return false
	}

	// 3) Read lines until we get a JSON array (filters logs/errors)
	var outVals []float64
	for {
		outLine, err := n.gpuProc.stdout.ReadBytes('\n')
		if err != nil {
			fmt.Println("❌ Read GPU stdout failed:", err)
			return false
		}
		str := strings.TrimSpace(string(outLine))
		if strings.HasPrefix(str, "[") {
			if err := json.Unmarshal([]byte(str), &outVals); err != nil {
				fmt.Printf("❌ JSON unmarshal failed (raw): '%s' [%v]\n", str, err)
				return false
			}
			break
		} else {
			// Print or log filtered lines for debugging
			fmt.Printf("[Go FILTERED LOG/ERROR from GPU]: %s\n", str)
		}
	}

	// 4) Sanity check + write back
	if len(outVals) != curr.Width*curr.Height {
		fmt.Printf("❌ Output size mismatch. Got %d, expected %d\n", len(outVals), curr.Width*curr.Height)
		return false
	}
	idx := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			curr.Neurons[y][x].Value = T(outVals[idx])
			idx++
		}
	}
	return true
}

func (n *Network[T]) forwardLayerWebGPUOLDMANUALHANDOVER(l int) bool {

	curr := n.Layers[l]

	// 1) Build payload
	type neuronJSON struct {
		Bias    float64   `json:"bias"`
		Inputs  []float64 `json:"inputs"`
		Weights []float64 `json:"weights"`
		Act     string    `json:"activation"`
	}
	layerData := make([]neuronJSON, 0, curr.Width*curr.Height)
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			cell := curr.Neurons[y][x]
			inp := make([]float64, len(cell.Inputs))
			wts := make([]float64, len(cell.Inputs))
			for i, c := range cell.Inputs {
				inp[i] = float64(any(n.Layers[c.SourceLayer].
					Neurons[c.SourceY][c.SourceX].Value).(T))
				wts[i] = float64(any(c.Weight).(T))
			}
			layerData = append(layerData, neuronJSON{
				Bias:    float64(any(cell.Bias).(T)),
				Inputs:  inp,
				Weights: wts,
				Act:     cell.Activation,
			})
		}
	}

	payload, err := json.Marshal(layerData)
	if err != nil {
		fmt.Println("❌ JSON marshal failed:", err)
		return false
	}

	// 2) Launch GPU process and pipe
	cmd := exec.Command("./gpu_forward_layer")
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		fmt.Println("❌ Failed to start gpu_forward_layer:", err)
		return false
	}
	stdin.Write(payload)
	stdin.Close()

	// 3) Capture all stdout
	outBytes, err := io.ReadAll(stdout)
	if err != nil {
		fmt.Println("❌ Read stdout failed:", err)
		return false
	}
	if err := cmd.Wait(); err != nil {
		fmt.Println("❌ gpu_forward_layer exited with error:", err)
		return false
	}

	// 4) Extract the JSON array of floats
	//    Matches “[” then digits, dots, commas, spaces, e/E/+/– then “]”
	re := regexp.MustCompile(`\[[0-9eE\+\-.,\s]+\]`)
	match := re.Find(outBytes)
	if match == nil {
		fmt.Println("❌ Couldn't find JSON array in output")
		return false
	}

	// 5) Unmarshal into a slice
	var outVals []float64
	if err := json.Unmarshal(match, &outVals); err != nil {
		fmt.Println("❌ JSON unmarshal failed:", err)
		return false
	}

	// 6) Sanity check + write back
	if len(outVals) != curr.Width*curr.Height {
		fmt.Println("❌ Output size mismatch")
		return false
	}
	idx := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			curr.Neurons[y][x].Value = T(outVals[idx])
			idx++
		}
	}

	return true
}

func (n *Network[T]) forwardLayerWebGPUWORKINGWITHJUSTFLOAT32ITHNK(l int) bool {
	curr := n.Layers[l]

	// 1) Build payload
	type neuronJSON struct {
		Bias    float64   `json:"bias"`
		Inputs  []float64 `json:"inputs"`
		Weights []float64 `json:"weights"`
		Act     string    `json:"activation"`
	}
	layerData := make([]neuronJSON, 0, curr.Width*curr.Height)
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			cell := curr.Neurons[y][x]
			inp := make([]float64, len(cell.Inputs))
			wts := make([]float64, len(cell.Inputs))
			for i, c := range cell.Inputs {
				inp[i] = float64(any(n.Layers[c.SourceLayer].
					Neurons[c.SourceY][c.SourceX].Value).(T))
				wts[i] = float64(any(c.Weight).(T))
			}
			layerData = append(layerData, neuronJSON{
				Bias:    float64(any(cell.Bias).(T)),
				Inputs:  inp,
				Weights: wts,
				Act:     cell.Activation,
			})
		}
	}

	payload, err := json.Marshal(layerData)
	if err != nil {
		fmt.Println("❌ JSON marshal failed:", err)
		return false
	}

	// 2) Launch GPU process and pipe
	cmd := exec.Command("./gpu_forward_layer")
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		fmt.Println("❌ Failed to start gpu_forward_layer:", err)
		return false
	}
	stdin.Write(payload)
	stdin.Close()

	// 3) Capture all stdout
	outBytes, err := io.ReadAll(stdout)
	if err != nil {
		fmt.Println("❌ Read stdout failed:", err)
		return false
	}
	if err := cmd.Wait(); err != nil {
		fmt.Println("❌ gpu_forward_layer exited with error:", err)
		return false
	}

	// 4) Extract the JSON array of floats
	//    Matches “[” then digits, dots, commas, spaces, e/E/+/– then “]”
	re := regexp.MustCompile(`\[[0-9eE\+\-.,\s]+\]`)
	match := re.Find(outBytes)
	if match == nil {
		fmt.Println("❌ Couldn't find JSON array in output")
		return false
	}

	// 5) Unmarshal into a slice
	var outVals []float64
	if err := json.Unmarshal(match, &outVals); err != nil {
		fmt.Println("❌ JSON unmarshal failed:", err)
		return false
	}

	// 6) Sanity check + write back
	if len(outVals) != curr.Width*curr.Height {
		fmt.Println("❌ Output size mismatch")
		return false
	}
	idx := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			curr.Neurons[y][x].Value = T(outVals[idx])
			idx++
		}
	}
	return true
}

func (n *Network[T]) forwardLayerWebGPUFILE(l int) bool {
	curr := n.Layers[l]
	inputSize := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			inputSize += len(curr.Neurons[y][x].Inputs)
		}
	}

	// Prepare data buffer
	type neuronInput struct {
		Bias    float64   `json:"bias"`
		Inputs  []float64 `json:"inputs"`
		Weights []float64 `json:"weights"`
		Act     string    `json:"activation"`
	}

	var layerData []neuronInput
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			neuron := curr.Neurons[y][x]
			var inputs []float64
			var weights []float64

			for _, c := range neuron.Inputs {
				val := float64(any(n.Layers[c.SourceLayer].Neurons[c.SourceY][c.SourceX].Value).(T))
				weight := float64(any(c.Weight).(T))
				inputs = append(inputs, val)
				weights = append(weights, weight)
			}

			layerData = append(layerData, neuronInput{
				Bias:    float64(any(neuron.Bias).(T)),
				Inputs:  inputs,
				Weights: weights,
				Act:     neuron.Activation,
			})
		}
	}

	// Write input JSON
	inputFile := fmt.Sprintf("layer%d_input.json", l)
	outputFile := fmt.Sprintf("layer%d_output.json", l)

	data, err := json.MarshalIndent(layerData, "", "  ")
	if err != nil {
		fmt.Println("❌ JSON marshal failed:", err)
		return false
	}
	if err := os.WriteFile(inputFile, data, 0644); err != nil {
		fmt.Println("❌ Write input file failed:", err)
		return false
	}

	// Call C++ binary
	cmd := exec.Command("./gpu_forward_layer", inputFile, outputFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println("❌ GPU binary execution failed:", err)
		fmt.Println("Output:", string(output))
		return false
	}

	// Read and parse output
	rawOut, err := os.ReadFile(outputFile)
	if err != nil {
		fmt.Println("❌ Read output file failed:", err)
		return false
	}

	var outputVals []float64
	if err := json.Unmarshal(rawOut, &outputVals); err != nil {
		fmt.Println("❌ Output unmarshal failed:", err)
		return false
	}

	// Write back to neuron values
	if len(outputVals) != curr.Width*curr.Height {
		fmt.Println("❌ Output size mismatch")
		return false
	}

	idx := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			curr.Neurons[y][x].Value = T(outputVals[idx])
			idx++
		}
	}
	return true
}
