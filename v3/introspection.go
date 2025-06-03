package paragon

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// MethodInfo represents metadata about a method, including its name, parameters, and parameter types.
type MethodInfo struct {
	MethodName string          `json:"method_name"`
	Parameters []ParameterInfo `json:"parameters"`
}

// ParameterInfo represents metadata about a parameter, including its name and type.
type ParameterInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// GetphaseMethodsJSON returns a JSON string containing all methods attached to the phase struct,
// including each method's parameters and their types.
func (n *Network[T]) GetphaseMethodsJSON() (string, error) {
	// Retrieve all methods and their metadata
	methods, err := n.GetphaseMethods()
	if err != nil {
		return "", fmt.Errorf("failed to retrieve methods: %w", err)
	}

	// Convert methods metadata to JSON
	data, err := json.MarshalIndent(methods, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to serialize methods to JSON: %w", err)
	}

	return string(data), nil
}

// GetphaseMethods retrieves all methods of the phase struct, including their names, parameters, and types.
func (n *Network[T]) GetphaseMethods() ([]MethodInfo, error) {
	var methods []MethodInfo

	// Use reflection to inspect the phase's methods
	bpType := reflect.TypeOf(n)
	for i := 0; i < bpType.NumMethod(); i++ {
		method := bpType.Method(i)

		// Collect parameter information for each method
		var params []ParameterInfo
		methodType := method.Type
		for j := 1; j < methodType.NumIn(); j++ { // Start from 1 to skip the receiver
			paramType := methodType.In(j)
			param := ParameterInfo{
				Name: fmt.Sprintf("param%d", j),
				Type: paramType.String(),
			}
			params = append(params, param)
		}

		// Append method information
		methods = append(methods, MethodInfo{
			MethodName: method.Name,
			Parameters: params,
		})
	}

	return methods, nil
}
