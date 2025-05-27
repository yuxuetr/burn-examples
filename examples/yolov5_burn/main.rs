#![allow(dead_code)] // Allow unused code for example simplicity & tests

// Declare modules
pub mod model;
pub mod utils;

// Necessary Imports
use burn::backend::wgpu::{WgpuDevice, Wgpu};
use burn::backend::Autodiff;
use burn::tensor::{Tensor, Distribution, Data, Shape, Float};
use burn::prelude::*;

use model::{YoloV5, YoloV5Config}; // YOLOv5 model
use utils::{
    BoundingBox, preprocess_image, parse_all_yolo_outputs, 
    non_maximum_suppression, get_yolov5s_anchors,
}; // YOLOv5 utils

use std::path::Path;
use image::DynamicImage; // For loading images
use clap::Parser; // For command-line arguments

// Command-line arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the input image
    #[clap(short, long)]
    image_path: Option<String>,

    /// Path to the model weights file (placeholder for now)
    #[clap(short, long)]
    weights_path: Option<String>,

    /// Confidence threshold for detections
    #[clap(short, long, default_value_t = 0.25)]
    conf_thres: f32,

    /// IoU threshold for Non-Maximum Suppression
    #[clap(short, long, default_value_t = 0.45)]
    iou_thres: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. Setup Backend
    type MyBackend = Wgpu; // Using WGPU backend
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::BestAvailable; // Selects the best available WGPU device

    println!("YOLOv5 Burn Example");
    println!("Using backend: WGPU ({:?})", device);
    println!("Confidence threshold: {}", args.conf_thres);
    println!("IoU threshold: {}", args.iou_thres);


    // 2. Model Initialization
    const NUM_CLASSES: usize = 80; // Example: COCO dataset (YOLOv5 is often trained on this)
    const NUM_ANCHORS_PER_SCALE: usize = 3; // Standard for YOLO
    const INPUT_WIDTH: u32 = 640;  // Typical YOLOv5 input width
    const INPUT_HEIGHT: u32 = 640; // Typical YOLOv5 input height

    let model_config = YoloV5Config::yolov5s(NUM_CLASSES, NUM_ANCHORS_PER_SCALE);
    let mut model: YoloV5<MyAutodiffBackend> = model_config.init(&device); // Make model mutable for potential weight loading

    println!(
        "YOLOv5s model initialized with {} classes and {} anchors per scale.",
        NUM_CLASSES, NUM_ANCHORS_PER_SCALE
    );
    println!("Network input dimensions: {}x{}", INPUT_WIDTH, INPUT_HEIGHT);

    // (Optional) Load weights
    if let Some(weights_file) = args.weights_path {
        println!("Attempting to load weights from: {} (Placeholder)", weights_file);
        // model = utils::load_yolov5_weights(model, &weights_file)
        //     .expect("Failed to load weights (implement actual loading)");
        println!("Note: Actual weight loading is not implemented in this example.");
    } else {
        println!("No weights file provided. Using randomly initialized model.");
    }

    // 3. Load and Preprocess Image
    let (image_tensor, original_img_dims): (Tensor<MyAutodiffBackend, 4>, (u32,u32)) = 
        if let Some(image_path_str) = args.image_path {
            println!("Loading image from: {}", image_path_str);
            let img_path = Path::new(&image_path_str);
            if !img_path.exists() {
                eprintln!("Error: Image path does not exist: {}", image_path_str);
                // Fallback to dummy tensor if image not found
                println!("Falling back to dummy random tensor.");
                let dummy_tensor = Tensor::<MyAutodiffBackend, 4>::random(
                    Shape::new([1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize]),
                    Distribution::Uniform(0.0, 1.0),
                    &device,
                );
                (dummy_tensor, (INPUT_WIDTH, INPUT_HEIGHT)) // Assuming dummy is already target size
            } else {
                let img = image::open(img_path)
                    .map_err(|e| format!("Failed to open image {}: {}", image_path_str, e))?;
                let original_dims = img.dimensions();
                println!("Original image dimensions: {}x{}", original_dims.0, original_dims.1);
                let tensor = preprocess_image(&img, (INPUT_WIDTH, INPUT_HEIGHT), &device)
                    .map_err(|e| format!("Failed to preprocess image: {}", e))?;
                (tensor, original_dims)
            }
        } else {
            println!("No image path provided. Using a dummy random tensor.");
            let dummy_tensor = Tensor::<MyAutodiffBackend, 4>::random(
                Shape::new([1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize]),
                Distribution::Uniform(0.0, 1.0), // Values between 0 and 1
                &device,
            );
            (dummy_tensor, (INPUT_WIDTH, INPUT_HEIGHT)) // Assuming dummy is already target size
        };
    
    println!("Input tensor prepared with shape: {:?}", image_tensor.dims());

    // 4. Define Anchors and Strides for YOLOv5s
    let anchors = get_yolov5s_anchors(); // Gets [[(w,h),...], [...], [...]]
    let strides = [8.0f32, 16.0f32, 32.0f32]; // Strides for P3, P4, P5
    println!("Anchors and strides defined for 3 scales (YOLOv5s).");

    // 5. Perform Forward Pass
    println!("Performing forward pass...");
    // The model forward pass should return 3 tensors, each with shape [B, NA, G, G, NC+5]
    // For Burn, if the model's forward directly returns a tuple of tensors, it's fine.
    // If it returns a struct containing tensors, adjust access accordingly.
    // Assuming model.forward() returns (Tensor, Tensor, Tensor) for the 3 scales.
    let (output_p3, output_p4, output_p5): (
        Tensor<MyAutodiffBackend, 5>,
        Tensor<MyAutodiffBackend, 5>,
        Tensor<MyAutodiffBackend, 5>,
    ) = model.forward(image_tensor);
    
    println!("Forward pass completed. Received 3 output tensors.");
    println!("Output P3 Shape: {:?}", output_p3.dims()); // [1, 3, 80, 80, 85]
    println!("Output P4 Shape: {:?}", output_p4.dims()); // [1, 3, 40, 40, 85]
    println!("Output P5 Shape: {:?}", output_p5.dims()); // [1, 3, 20, 20, 85]

    // 6. Parse Outputs
    println!("Parsing outputs from all scales...");
    let all_predictions: Vec<BoundingBox> = parse_all_yolo_outputs(
        (output_p3, output_p4, output_p5),
        &anchors,
        NUM_CLASSES,
        args.conf_thres, // Use confidence threshold from args
        &strides,
        original_img_dims,      // Original image dimensions
        (INPUT_WIDTH, INPUT_HEIGHT) // Network input dimensions
    );

    println!(
        "Total raw predictions from all scales (before NMS): {}",
        all_predictions.len()
    );

    // 7. Apply Non-Maximum Suppression (NMS)
    println!(
        "Applying Non-Maximum Suppression with IoU threshold: {}", args.iou_thres
    );
    // The NMS function in utils.rs takes `confidence_threshold` as an argument,
    // but `parse_all_yolo_outputs` already filters by confidence.
    // Pass a low threshold (e.g., 0.0) to NMS if pre-filtered, or ensure NMS's threshold is also args.conf_thres.
    // For simplicity, let's assume parse_all_yolo_outputs did the primary confidence filtering.
    let final_boxes = non_maximum_suppression(
        all_predictions,
        args.iou_thres, // Use IoU threshold from args
        0.01, // A small confidence threshold for NMS, as primary filtering is done in parsing.
              // Or, ensure parse_all_yolo_outputs doesn't filter by conf, and do it here/in NMS.
              // Current utils.rs parse_yolo_output *does* filter by confidence.
    );

    // 8. Print Results
    println!("----------------------------------------");
    println!("Number of predictions after NMS: {}", final_boxes.len());
    println!("----------------------------------------");

    if final_boxes.is_empty() {
        println!("No objects detected after NMS.");
    } else {
        println!("Detected objects (original image coordinates):");
        for (i, bbox) in final_boxes.iter().enumerate() {
            println!(
                "  Box {}: Class ID {}, Confidence {:.2}%, Coords (center_x, center_y, w, h): ({:.1}, {:.1}, {:.1}, {:.1})",
                i + 1,
                bbox.class_id,
                bbox.confidence * 100.0,
                bbox.x_center,
                bbox.y_center,
                bbox.width,
                bbox.height
            );
            // To draw on image: convert center_x, center_y, w, h to x_min, y_min, x_max, y_max
            // let x_min = bbox.x_center - bbox.width / 2.0;
            // let y_min = bbox.y_center - bbox.height / 2.0;
            // ... then use a drawing library.
        }
    }
    println!("----------------------------------------");
    println!("YOLOv5 Burn Example Finished.");

    Ok(())
}

// This line ensures that the tests defined in tests.rs are compiled and run.
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
