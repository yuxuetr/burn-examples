# YOLOv3 Object Detection Model in Burn

## Purpose

This example implements the YOLOv3 (You Only Look Once, version 3) object detection model using the [Burn](https://burn.dev) deep learning framework. It aims to demonstrate how a complex computer vision model like YOLOv3 can be built with Burn, showcasing its capabilities for defining custom architectures and handling tensor operations.

## Features

*   **Darknet-53 Backbone**: Implements the convolutional neural network used as the feature extractor in YOLOv3.
*   **YOLO Detection Heads**: Includes detection heads for three different scales, allowing detection of objects of various sizes.
*   **Output Parsing**: Basic utility to decode the raw tensor outputs from the model into bounding box predictions.
*   **Non-Maximum Suppression (NMS)**: Utility function to filter overlapping bounding boxes and retain the most confident detections.
*   **Modular Design**: Separates model definition, utility functions, and the main execution logic into different files.

## Current Status / Limitations

*   **Dummy Input**: The `main.rs` currently uses a dummy input tensor (all zeros) for demonstration purposes. Actual image loading and preprocessing are not implemented.
*   **No Pre-trained Weights**: The functionality to load pre-trained weights from Darknet `.weights` files (`utils::load_darknet_weights`) is a placeholder. The model runs with randomly initialized weights, so it will not produce meaningful object detections.
*   **Basic Output Parsing**: The output parsing in `utils.rs` is a basic implementation and might need further refinement for robust real-world use.
*   **WGPU Backend Focus**: The example is primarily set up to use the WGPU backend.

## How to Run

1.  Navigate to the example directory:
    ```bash
    cd examples/yolov3_burn
    ```
2.  Run the example using Cargo:
    ```bash
    cargo run
    ```
    This will execute the `main` function in `main.rs`, which initializes the model, performs a forward pass with a dummy input, parses the output, applies NMS, and prints the (currently non-meaningful) results.

    Ensure you have the necessary WGPU dependencies installed on your system if you haven't used Burn with WGPU before. Refer to the Burn documentation for backend setup.

## How to Test

1.  Navigate to the example directory:
    ```bash
    cd examples/yolov3_burn
    ```
2.  Run the tests using Cargo:
    ```bash
    cargo test
    ```
    This will execute the unit and integration tests defined in `tests.rs`.

## Code Structure

The example is organized into several key files:

*   `main.rs`: Contains the main application logic. It sets up the backend, initializes the YOLOv3 model, creates a dummy input, performs a forward pass, parses the output, applies NMS, and prints the results.
*   `model.rs`: Defines the YOLOv3 model architecture, including `ConvBlock`, `ResidualBlock`, and the main `YoloV3` struct.
*   `utils.rs`: Provides utility functions such as `parse_yolo_output` for decoding model predictions and `non_maximum_suppression` (NMS) for filtering bounding boxes. It also contains a placeholder for `load_darknet_weights`.
*   `tests.rs`: Includes unit tests for individual components (like `ConvBlock`, `ResidualBlock`), model initialization, forward pass shape correctness, output parsing logic, and NMS functionality.
*   `Cargo.toml`: Specifies the dependencies for this specific YOLOv3 example, primarily Burn and its features.
*   `README.md`: This file, providing an overview and instructions for the example.
