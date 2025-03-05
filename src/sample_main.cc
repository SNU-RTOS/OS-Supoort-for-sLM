#include <iostream>
#include <vector>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

int main(int argc, char* argv[]) {
  // Check if the model path is provided
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_model_path>\n";
    return 1;
  }

  // ===================================================================
  // 1. MODEL LOADING
  // ===================================================================
  // Key functions to understand:
  // - FlatBufferModel::BuildFromFile: Loads model from file using FlatBuffers
  // Key files:
  // - tensorflow/lite/core/model_builder.h, model_buidler.cc: FlatBufferModel implementation
  // - tensorflow/lite/schema/schema.fbs: Defines the FlatBuffer schema
  std::string model_path = argv[1];
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model) {
    std::cerr << "Failed to load model at " << model_path << std::endl;
    return 1;
  }
  std::cout << "Model loaded successfully." << std::endl;

  // ===================================================================
  // 2. INTERPRETER SETUP
  // ===================================================================
  // Key functions to understand:
  // - InterpreterBuilder::InterpreterBuilder: Sets up the interpreter builder
  // - InterpreterBuilder::operator(): Builds the interpreter instance
  // - OpResolver: Provides op implementations to the interpreter
  // Key files:
  // - tensorflow/lite/interpreter_builder.h: InterpreterBuilder implementation
  // - tensorflow/lite/kernels/register.h: BuiltinOpResolver implementation

  // Create the op resolver
  // This maps TFLite operations to their implementations
  tflite::ops::builtin::BuiltinOpResolver resolver;

  // Build the interpreter builder
  tflite::InterpreterBuilder builder(*model, resolver);

  // Build the interpreter
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
    return 1;
  }
  std::cout << "Interpreter built successfully." << std::endl;

  // ===================================================================
  // 3. TENSOR ALLOCATION
  // ===================================================================
  // Key functions to understand:
  // - Interpreter::AllocateTensors: Allocates memory for all tensors
  // - Interpreter::ResizeInputTensor: Changes input dimensions if needed
  // Key files:
  // - tensorflow/lite/core/interpreter.h, interpreter.cc: Main interpreter implementation
  // - tensorflow/lite/core/subgraph.h, subgraph.cc: Core execution implementation
  // - tensorflow/lite/arena_planner.h, arena_planner.cc: Memory allocation planning
  
  // Allocate memory for all tensors that are not model parameters which are memory-mapped
  // This must be called before inference
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    return 1;
  }
  std::cout << "Tensors allocated successfully." << std::endl;

  // ===================================================================
  // 4. INFERENCE
  // ===================================================================
  // Key functions to understand:
  // - Interpreter::Invoke: Runs the model
  // - Subgraph::Invoke: The actual implementation of model execution
  // Key files:
  // - tensorflow/lite/core/interpreter.h, interpreter.cc: Invoke interface
  // - tensorflow/lite/core/subgraph.h, subgraph.cc: Core execution logic


  // 4-1. Input Setup
  // Get input tensor information
  const auto& input_tensor = interpreter->input_tensor(0);
  const auto& input_dims = input_tensor->dims;

  // Get a pointer to the input tensor data
  // The type must match the tensor's type (here we assume float32)
  float* input = interpreter->typed_input_tensor<float>(0);
  
  // Fill input tensor with data
  // In a real application, this would come from your input source
  std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (int i = 0; i < input_data.size(); i++) {
    input[i] = input_data[i];
  }
  std::cout << "Input data set." << std::endl;

  // 4-2. Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Failed to invoke interpreter." << std::endl;
    return 1;
  }
  std::cout << "Inference completed successfully." << std::endl;


  // 4-3. Output extraction  
  // Get output tensor information
  const auto& output_tensor = interpreter->output_tensor(0);
  const auto& output_dims = output_tensor->dims;
  
  std::cout << "Output tensor type: " << TfLiteTypeGetName(output_tensor->type) << std::endl;
  std::cout << "Output tensor shape: ";
  for (int j = 0; j < output_dims->size; j++) {
    std::cout << output_dims->data[j] << " ";
  }
  std::cout << std::endl;
  
  // Calculate total number of elements in the output tensor
  int output_size = 1;
  for (int i = 0; i < output_dims->size; i++) {
    output_size *= output_dims->data[i];
  }

  // Get a pointer to the output tensor data
  float* output = interpreter->typed_output_tensor<float>(0);

  // Process output data
  std::cout << "Output data:" << std::endl;
  for (int i = 0; i < output_size; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  // The interpreter and model are automatically cleaned up
  // when their unique_ptr goes out of scope
  return 0;
}