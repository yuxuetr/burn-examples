#![allow(dead_code)] // Allow unused code for example simplicity & tests

// Declare modules
pub mod model;
pub mod utils;

// Necessary Imports
use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::AutoGraphicsApi; // For Wgpu backend
use burn::tensor::Tensor;
use model::{YoloV3, YoloV3Config}; // Assuming YoloV3 & YoloV3Config are pub in model.rs
use utils::BoundingBox; // Assuming BoundingBox is pub in utils.rs

fn main() {
    // 1. Setup Backend
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = Default::default(); // Default WGPU device

    println!("YOLOv3 Burn Example");
    println!("Using backend: {:?}", std::any::type_name::<MyAutodiffBackend>());

    // 2. Model Initialization
    const NUM_CLASSES: usize = 80; // Example: COCO dataset
    const NUM_ANCHORS_PER_SCALE: usize = 3;
    const INPUT_WIDTH: u32 = 416;
    const INPUT_HEIGHT: u32 = 416;

    let model_config = YoloV3Config::new(NUM_CLASSES, NUM_ANCHORS_PER_SCALE);
    let mut model: YoloV3<MyAutodiffBackend> = model_config.init();
    model.to_device(&device); // Move model to device

    println!("Model initialized with {} classes and {} anchors per scale.", NUM_CLASSES, NUM_ANCHORS_PER_SCALE);
    println!("Input dimensions: {}x{}", INPUT_WIDTH, INPUT_HEIGHT);


    // (Optional) Load weights - for now, we use the initialized (random) weights
    // match utils::load_darknet_weights(&mut model, "path/to/yolov3.weights") {
    //     Ok(_) => println!("Weights loaded successfully (placeholder)."),
    //     Err(e) => eprintln!("Error loading weights (placeholder): {}", e),
    // }


    // 3. Load Example Image (Simulated - Dummy Input Tensor)
    let dummy_input_tensor = Tensor::<MyAutodiffBackend, 4>::zeros(
        [1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize],
    ).to_device(&device);
    println!("Dummy input tensor created with shape: [1, 3, {}, {}]", INPUT_HEIGHT, INPUT_WIDTH);

    // 4. Define Anchors (COCO anchors scaled for 416x416 input)
    // Format: (width, height) relative to network input size (e.g., 416x416)
    // These are typically derived from the dataset the model was trained on.
    let anchors_scale1: Vec<(f32, f32)> = vec![(116., 90.), (156., 198.), (373., 326.)]; // For large objects (small grid: 13x13)
    let anchors_scale2: Vec<(f32, f32)> = vec![(30., 61.), (62., 45.), (59., 119.)];   // For medium objects (medium grid: 26x26)
    let anchors_scale3: Vec<(f32, f32)> = vec![(10., 13.), (16., 30.), (33., 23.)];    // For small objects (large grid: 52x52)
    println!("Anchors defined for 3 scales.");

    // 5. Perform Forward Pass
    println!("Performing forward pass...");
    let (output_scale1, output_scale2, output_scale3) = model.forward(dummy_input_tensor);
    println!("Forward pass completed. Received 3 output tensors.");
    // Optionally print shapes:
    // println!("Output Scale 1 Shape: {:?}", output_scale1.dims());
    // println!("Output Scale 2 Shape: {:?}", output_scale2.dims());
    // println!("Output Scale 3 Shape: {:?}", output_scale3.dims());


    // 6. Parse Outputs
    let confidence_threshold = 0.5; // Example confidence threshold
    let mut all_predictions: Vec<BoundingBox> = Vec::new();

    println!("Parsing output from Scale 1 (large objects)...");
    let boxes_scale1 = utils::parse_yolo_output(
        output_scale1,
        &anchors_scale1,
        NUM_CLASSES,
        confidence_threshold,
        (INPUT_WIDTH, INPUT_HEIGHT),
    );
    all_predictions.extend(boxes_scale1);

    println!("Parsing output from Scale 2 (medium objects)...");
    let boxes_scale2 = utils::parse_yolo_output(
        output_scale2,
        &anchors_scale2,
        NUM_CLASSES,
        confidence_threshold,
        (INPUT_WIDTH, INPUT_HEIGHT),
    );
    all_predictions.extend(boxes_scale2);

    println!("Parsing output from Scale 3 (small objects)...");
    let boxes_scale3 = utils::parse_yolo_output(
        output_scale3,
        &anchors_scale3,
        NUM_CLASSES,
        confidence_threshold,
        (INPUT_WIDTH, INPUT_HEIGHT),
    );
    all_predictions.extend(boxes_scale3);

    println!("Total raw predictions from all scales (before NMS): {}", all_predictions.len());

    // 7. Apply Non-Maximum Suppression (NMS)
    let iou_threshold = 0.45; // Example IoU threshold for NMS
    println!("Applying Non-Maximum Suppression with IoU threshold: {} and Confidence threshold: {}", iou_threshold, confidence_threshold);
    let final_boxes = utils::non_maximum_suppression(
        all_predictions, // NMS function expects Vec<BoundingBox>
        iou_threshold,
        confidence_threshold, // NMS function also has confidence thresholding, could be redundant if parse_yolo_output already filters
    );

    // 8. Print Results
    println!("----------------------------------------");
    println!("Number of predictions after NMS: {}", final_boxes.len());
    println!("----------------------------------------");

    if final_boxes.is_empty() {
        println!("No objects detected after NMS.");
    } else {
        println!("Detected objects:");
        for (i, bbox) in final_boxes.iter().enumerate() {
            println!(
                "  Box {}: Class ID {}, Confidence {:.2}%, Coords (x,y,w,h): ({:.1}, {:.1}, {:.1}, {:.1})",
                i + 1,
                bbox.class_id,
                bbox.confidence * 100.0,
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height
            );
        }
    }
    println!("----------------------------------------");
    println!("YOLOv3 Burn Example Finished.");
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
