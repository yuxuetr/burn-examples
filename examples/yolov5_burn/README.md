# YOLOv5 Example in Burn

## Overview

This example demonstrates an implementation of the YOLOv5 object detection model using the [Burn](https://burn.dev) deep learning framework in Rust. It provides a basic structure for running inference with a YOLOv5 model.

## Prerequisites

*   **Rust and Cargo:** Ensure you have Rust and Cargo installed. You can find installation instructions at [rust-lang.org](https://www.rust-lang.org/tools/install).
*   **WGPU Backend Dependencies:** Burn's WGPU backend might require system dependencies for graphics APIs (e.g., Vulkan, Metal, DirectX). Please refer to the Burn documentation or `wgpu` crate for specific requirements for your platform.

## How to Run

1.  **Navigate to the example directory:**
    ```bash
    cd examples/yolov5_burn
    ```

2.  **Run using Cargo:**
    ```bash
    cargo run --release -- [OPTIONS]
    ```
    It's recommended to run in `release` mode for better performance.

    **Command-Line Arguments:**

    *   `--image-path <PATH_TO_IMAGE>`: (Optional) Path to the input image file (e.g., `my_image.jpg`). If not provided, a dummy random tensor will be used as input.
    *   `--weights-path <PATH_TO_WEIGHTS>`: (Optional) Path to the pre-trained model weights file. **Note: This is currently a placeholder.** The example will run with randomly initialized weights if not specified or if the loading logic is not fully implemented.
    *   `--conf-thres <THRESHOLD>`: (Optional) Confidence threshold for filtering detections. Default: `0.25`.
    *   `--iou-thres <THRESHOLD>`: (Optional) IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS). Default: `0.45`.

    **Example Command:**
    ```bash
    # Using a dummy tensor (no image or weights)
    cargo run --release

    # With an image (using random weights as weight loading is a placeholder)
    cargo run --release -- --image-path ./path/to/your/image.jpg

    # With an image and custom thresholds
    cargo run --release -- --image-path ./path/to/your/image.jpg --conf-thres 0.3 --iou-thres 0.5
    ```

## Functionality

The example performs the following steps:
1.  Initializes the YOLOv5 model (YOLOv5s architecture by default).
2.  If an image path is provided, it loads the image, preprocesses it (resize, letterbox, normalize), and converts it into a tensor. Otherwise, it uses a dummy random tensor.
3.  Performs a forward pass of the model with the input tensor.
4.  Parses the raw output from the model's detection heads.
5.  Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
6.  Prints the final detected objects to the console, including class ID, confidence score, and bounding box coordinates.

## Model Details

*   **Architecture:** The example implements the YOLOv5s (small) model architecture.
*   **Components:**
    *   **Backbone:** A CSPDarknet-style backbone (as described in YOLOv4/YOLOv5) for feature extraction.
    *   **Neck:** A PANet (Path Aggregation Network) structure for feature fusion across different scales.
    *   **Detection Heads:** YOLO-style detection heads responsible for predicting bounding boxes, objectness scores, and class probabilities at three different feature map scales.

## Weight Loading Note

The function `load_yolov5_weights` in `utils.rs` is currently a **placeholder**. To use this example with pre-trained YOLOv5 weights (which are typically in PyTorch's `.pt` format or ONNX), you would need to:
1.  **Convert the weights:** Convert the original YOLOv5 weights into a format that Burn can readily consume (e.g., `.npz` files, or a custom binary format). This often involves writing a separate Python script using PyTorch to extract and save the weights.
2.  **Implement the loading logic:** Update the `load_yolov5_weights` function in `utils.rs` to correctly read your converted weight files and assign them to the corresponding layers in the Burn model. This requires careful mapping of layer names and weight tensor shapes.

Without implementing these steps, the model will run with randomly initialized weights, and its detections will not be meaningful.

## Structure

The example is organized into the following key files:

*   `main.rs`: The main application entry point. Handles command-line arguments, model initialization, image loading (optional), inference, and output printing.
*   `model.rs`: Defines the YOLOv5 neural network architecture, including all its modules (e.g., `ConvBlock`, `CSPStage`, `SPPF`, `YoloV5Backbone`, `YoloV5Neck`, `DetectionHead`, `YoloV5`).
*   `utils.rs`: Contains utility functions for tasks such as image preprocessing (`preprocess_image`), output parsing (`parse_yolo_output`, `parse_all_yolo_outputs`), Non-Maximum Suppression (`non_maximum_suppression`), anchor definitions (`get_yolov5s_anchors`), and the placeholder for weight loading (`load_yolov5_weights`).
*   `tests.rs`: Includes unit tests for various components of the model and utility functions to ensure their correctness.
*   `README.md`: This file, providing information about the example.
