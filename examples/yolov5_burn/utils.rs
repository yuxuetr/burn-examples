use crate::model::YoloV5; // Updated import
use burn::{
    module::Module,
    nn::sigmoid, // For sigmoid activation in output parsing
    tensor::{backend::Backend, Tensor, Data, Shape, Int, Float}, // For tensor operations and types
    prelude::*,
};
use image::{DynamicImage, GenericImageView, imageops::FilterType}; // For image preprocessing

// YOLOv5s Anchors (for 3 output scales)
// These are typically defined for specific input sizes (e.g., 640x640)
// Format: [(width1, height1), (width2, height2), ...]
// Order corresponds to P3, P4, P5 feature maps (smallest to largest stride)
const ANCHORS_YOLOV5S_P3: &[(f32, f32); 3] = &[(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)];
const ANCHORS_YOLOV5S_P4: &[(f32, f32); 3] = &[(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)];
const ANCHORS_YOLOV5S_P5: &[(f32, f32); 3] = &[(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)];

pub fn get_yolov5s_anchors() -> Vec<Vec<(f32, f32)>> {
    vec![
        ANCHORS_YOLOV5S_P3.to_vec(),
        ANCHORS_YOLOV5S_P4.to_vec(),
        ANCHORS_YOLOV5S_P5.to_vec(),
    ]
}


#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
  pub x_center: f32, // Center x
  pub y_center: f32, // Center y
  pub width: f32,
  pub height: f32,
  pub class_id: usize,
  pub confidence: f32,
}

// Calculate Intersection over Union (IoU)
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let b1_x1 = box1.x_center - box1.width / 2.0;
    let b1_y1 = box1.y_center - box1.height / 2.0;
    let b1_x2 = box1.x_center + box1.width / 2.0;
    let b1_y2 = box1.y_center + box1.height / 2.0;

    let b2_x1 = box2.x_center - box2.width / 2.0;
    let b2_y1 = box2.y_center - box2.height / 2.0;
    let b2_x2 = box2.x_center + box2.width / 2.0;
    let b2_y2 = box2.y_center + box2.height / 2.0;

    let x1_inter = b1_x1.max(b2_x1);
    let y1_inter = b1_y1.max(b2_y1);
    let x2_inter = b1_x2.min(b2_x2);
    let y2_inter = b1_y2.min(b2_y2);

    let inter_width = (x2_inter - x1_inter).max(0.0);
    let inter_height = (y2_inter - y1_inter).max(0.0);
    let inter_area = inter_width * inter_height;

    let box1_area = box1.width * box1.height;
    let box2_area = box2.width * box2.height;

    let union_area = box1_area + box2_area - inter_area;

    if union_area == 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

// Non-Maximum Suppression (NMS)
pub fn non_maximum_suppression(
  mut predictions: Vec<BoundingBox>,
  iou_threshold: f32,
  confidence_threshold: f32, // This threshold is applied *before* calling NMS in typical YOLO pipelines
) -> Vec<BoundingBox> {
  // Filter by confidence already done if predictions come from parse_yolo_outputs
  predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

  let mut selected_boxes: Vec<BoundingBox> = Vec::new();
  let mut active = vec![true; predictions.len()];

  for i in 0..predictions.len() {
    if active[i] {
      selected_boxes.push(predictions[i]);
      for j in (i + 1)..predictions.len() {
        if active[j] && calculate_iou(&predictions[i], &predictions[j]) > iou_threshold {
          active[j] = false;
        }
      }
    }
  }
  selected_boxes
}


// Parse YOLOv5 model output
// Output tensor shape: [batch_size, num_anchors, grid_h, grid_w, (5 + num_classes)]
// (tx, ty, tw, th, conf, class_probs...)
pub fn parse_yolo_output<B: Backend>(
    output_tensor: Tensor<B, 5>, // Expecting 5D tensor now
    scale_anchors: &[(f32, f32)], // Anchors for this specific scale/grid
    num_classes: usize,
    confidence_threshold: f32,
    stride: f32, // Stride of the current feature map (e.g., 8, 16, 32)
    original_img_dims: (u32, u32), // (original_width, original_height)
    network_input_dims: (u32, u32), // (network_input_width, network_input_height)
) -> Vec<BoundingBox> {
    let mut bboxes: Vec<BoundingBox> = Vec::new();
    let [batch_size, num_scale_anchors, grid_h, grid_w, num_attrs] = output_tensor.dims();
    
    if num_attrs != (5 + num_classes) {
        panic!("Output tensor attributes mismatch num_classes + 5");
    }

    // YOLOv5 applies sigmoid to xy, wh, and confidence
    let output_tensor_sigmoid = sigmoid(output_tensor);
    let output_data = output_tensor_sigmoid.into_data().convert::<f32>().value;

    // Calculate scaling factors for letterboxing/padding
    let (net_w, net_h) = network_input_dims;
    let (orig_w, orig_h) = original_img_dims;
    let gain = (net_w as f32 / orig_w as f32).min(net_h as f32 / orig_h as f32); // gain  = old / new
    let pad_x = (net_w as f32 - orig_w as f32 * gain) / 2.0;
    let pad_y = (net_h as f32 - orig_h as f32 * gain) / 2.0;


    for b_idx in 0..batch_size { // Should typically be 1 for inference
        for anchor_idx in 0..num_scale_anchors {
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let base_idx = b_idx * (num_scale_anchors * grid_h * grid_w * num_attrs) +
                                   anchor_idx * (grid_h * grid_w * num_attrs) +
                                   gy * (grid_w * num_attrs) +
                                   gx * num_attrs;

                    let tx = output_data[base_idx];
                    let ty = output_data[base_idx + 1];
                    let tw = output_data[base_idx + 2];
                    let th = output_data[base_idx + 3];
                    let conf = output_data[base_idx + 4];

                    if conf < confidence_threshold {
                        continue;
                    }

                    // Class probabilities
                    let mut class_id = 0;
                    let mut max_class_prob = 0.0;
                    for c_idx in 0..num_classes {
                        let class_prob = output_data[base_idx + 5 + c_idx];
                        if class_prob > max_class_prob {
                            max_class_prob = class_prob;
                            class_id = c_idx;
                        }
                    }
                    
                    let final_confidence = conf * max_class_prob; // Multiply object confidence by class confidence
                    if final_confidence < confidence_threshold {
                        continue;
                    }

                    // Decode bounding box
                    // YOLOv5 formula:
                    // bx = (2 * sigmoid(tx) - 0.5 + gx) * stride
                    // by = (2 * sigmoid(ty) - 0.5 + gy) * stride
                    // bw = (2 * sigmoid(tw))^2 * anchor_w
                    // bh = (2 * sigmoid(th))^2 * anchor_h
                    // Note: sigmoid is already applied to tx, ty, tw, th from output_tensor_sigmoid

                    let pred_x = (2.0 * tx - 0.5 + gx as f32) * stride;
                    let pred_y = (2.0 * ty - 0.5 + gy as f32) * stride;
                    
                    let anchor_w = scale_anchors[anchor_idx].0;
                    let anchor_h = scale_anchors[anchor_idx].1;

                    let pred_w = (2.0 * tw).powi(2) * anchor_w;
                    let pred_h = (2.0 * th).powi(2) * anchor_h;

                    // Adjust coordinates back to original image dimensions (remove letterbox/padding)
                    let x_center = (pred_x - pad_x) / gain;
                    let y_center = (pred_y - pad_y) / gain;
                    let width = pred_w / gain;
                    let height = pred_h / gain;
                    
                    // Clip to image boundaries (optional, but good practice)
                    // let x_center = x_center.clamp(0.0, orig_w as f32);
                    // let y_center = y_center.clamp(0.0, orig_h as f32);
                    // let width = width.clamp(0.0, orig_w as f32);
                    // let height = height.clamp(0.0, orig_h as f32);


                    bboxes.push(BoundingBox {
                        x_center,
                        y_center,
                        width,
                        height,
                        class_id,
                        confidence: final_confidence,
                    });
                }
            }
        }
    }
    bboxes
}

// Helper to parse all output scales and aggregate results
pub fn parse_all_yolo_outputs<B: Backend>(
    outputs: (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>), // (P3, P4, P5 outputs)
    all_anchors: &[Vec<(f32, f32)>], // Anchors for P3, P4, P5
    num_classes: usize,
    confidence_threshold: f32,
    strides: &[f32; 3], // e.g., [8.0, 16.0, 32.0]
    original_img_dims: (u32, u32),
    network_input_dims: (u32, u32),
) -> Vec<BoundingBox> {
    let mut all_bboxes = Vec::new();

    // P3 output
    let bboxes_p3 = parse_yolo_output(
        outputs.0, &all_anchors[0], num_classes, confidence_threshold, strides[0], 
        original_img_dims, network_input_dims
    );
    all_bboxes.extend(bboxes_p3);

    // P4 output
    let bboxes_p4 = parse_yolo_output(
        outputs.1, &all_anchors[1], num_classes, confidence_threshold, strides[1],
        original_img_dims, network_input_dims
    );
    all_bboxes.extend(bboxes_p4);

    // P5 output
    let bboxes_p5 = parse_yolo_output(
        outputs.2, &all_anchors[2], num_classes, confidence_threshold, strides[2],
        original_img_dims, network_input_dims
    );
    all_bboxes.extend(bboxes_p5);
    
    all_bboxes
}


// Image Preprocessing: Letterbox resize and normalize
pub fn preprocess_image<B: Backend>(
    img: &DynamicImage,
    input_size: (u32, u32), // (width, height) for the network
    device: &Device<B>,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let (orig_w, orig_h) = img.dimensions();
    let (target_w, target_h) = input_size;

    // Calculate aspect ratio and scaling factor
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    // Resize image while maintaining aspect ratio
    let resized_img = img.resize_exact(new_w, new_h, FilterType::Triangle); // Triangle is a good default

    // Create a new image with padding
    let mut letterboxed_img = image::RgbImage::new(target_w, target_h);
    for pixel in letterboxed_img.pixels_mut() {
        *pixel = image::Rgb([114, 114, 114]); // YOLOv5 common padding color (grey)
    }

    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;
    image::imageops::overlay(&mut letterboxed_img, &resized_img.to_rgb8(), pad_x as i64, pad_y as i64);
    
    // Convert to tensor [1, C, H, W] and normalize to 0.0-1.0
    let mut img_data: Vec<f32> = Vec::with_capacity((target_w * target_h * 3) as usize);
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = letterboxed_img.get_pixel(x, y);
            img_data.push(pixel[0] as f32 / 255.0); // R
            img_data.push(pixel[1] as f32 / 255.0); // G
            img_data.push(pixel[2] as f32 / 255.0); // B
        }
    }
    
    // Transpose from [H, W, C] to [C, H, W]
    let mut chw_data: Vec<f32> = Vec::with_capacity(img_data.len());
    for c_idx in 0..3 { // R, G, B
        for h_idx in 0..target_h {
            for w_idx in 0..target_w {
                chw_data.push(img_data[(h_idx * target_w * 3 + w_idx * 3 + c_idx) as usize]);
            }
        }
    }

    let shape = Shape::new([1, 3, target_h as usize, target_w as usize]);
    let tensor_data = Data::new(chw_data, shape.clone());
    let tensor = Tensor::<B, 4>::from_data(tensor_data.convert(), device);
    
    Ok(tensor)
}


// Placeholder for Weight Loading Function
// This is a complex task and typically requires a separate weight conversion script
// from the original format (e.g., .pt files for PyTorch) to a Burn-compatible format.
pub fn load_yolov5_weights<B: Backend>(
  model: YoloV5<B>, // Model instance to load weights into
  _filepath: &str,   // Path to the pre-trained weights file (e.g., a .npz or custom format)
) -> Result<YoloV5<B>, Box<dyn std::error::Error>> {
  println!("Placeholder: Attempting to load YOLOv5 weights.");
  println!("Actual weight loading requires a specific format and mapping logic.");
  println!("This function needs to be implemented based on how weights are stored.");
  
  // Example of how one might load weights for a specific layer if they were available
  // let conv_weights = Tensor::<B, 4>::random([out_c, in_c, k, k], burn::tensor::Distribution::Normal(0.0, 0.02), &device);
  // let bn_gamma = Tensor::<B, 1>::ones([out_c], &device);
  // let bn_beta = Tensor::<B, 1>::zeros([out_c], &device);
  // let bn_running_mean = Tensor::<B, 1>::zeros([out_c], &device);
  // let bn_running_var = Tensor::<B, 1>::ones([out_c], &device);
  //
  // model.backbone.stem.conv.weight.load_record(conv_weights.into_record());
  // model.backbone.stem.bn.gamma.load_record(bn_gamma.into_record());
  // ... and so on for all layers and parameters.

  // For now, return the initialized model without loading weights
  Ok(model)
}
