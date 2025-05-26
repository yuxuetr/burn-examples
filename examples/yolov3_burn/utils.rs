use burn::tensor::{backend::Backend, Tensor, Data, Shape, ElementConversion};
use burn::tensor::activation::{sigmoid, softmax}; // Added for parse_yolo_output
use crate::model::YoloV3; // For load_darknet_weights signature

// 2. Bounding Box Struct
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x: f32,          // Center x or top-left x (be consistent)
    pub y: f32,          // Center y or top-left y
    pub width: f32,
    pub height: f32,
    pub class_id: usize,
    pub confidence: f32,
}

// Helper function for NMS
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    // Convert to top-left and bottom-right coordinates if they are center x,y
    // Assuming x,y are top-left for simplicity here. If they are center, conversion is needed:
    // let box1_x1 = box1.x - box1.width / 2.0;
    // let box1_y1 = box1.y - box1.height / 2.0;
    // let box1_x2 = box1.x + box1.width / 2.0;
    // let box1_y2 = box1.y + box1.height / 2.0;
    // Same for box2. For now, assuming x, y are top-left.

    let x1_inter = box1.x.max(box2.x);
    let y1_inter = box1.y.max(box2.y);
    let x2_inter = (box1.x + box1.width).min(box2.x + box2.width);
    let y2_inter = (box1.y + box1.height).min(box2.y + box2.height);

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

// 3. Non-Maximum Suppression (NMS) Function
pub fn non_maximum_suppression(
    mut predictions: Vec<BoundingBox>,
    iou_threshold: f32,
    confidence_threshold: f32,
) -> Vec<BoundingBox> {
    // Filter out boxes below confidence_threshold
    predictions.retain(|bbox| bbox.confidence >= confidence_threshold);

    // Sort remaining boxes by confidence score in descending order
    predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected_boxes: Vec<BoundingBox> = Vec::new();
    let mut i = 0;
    while i < predictions.len() {
        let current_box = predictions[i];
        selected_boxes.push(current_box);
        i += 1; // Move to the next box to consider

        // Remove boxes that have high IoU with the current_box
        // Need to iterate backwards or adjust indices carefully if removing in place from `predictions`
        // A simpler way is to build a new list of boxes to keep for the next iteration,
        // but since `predictions` is already sorted and we only look forward,
        // we can just mark them for removal or build a new predictions list.
        // For this implementation, let's filter `predictions` based on IoU with `current_box`.
        // This means boxes processed after `current_box` are checked against it.
        
        let mut j = i; // Start checking from the box after the one just added
        while j < predictions.len() {
            if calculate_iou(&current_box, &predictions[j]) > iou_threshold {
                predictions.remove(j); // This modifies the list, so j should not be incremented
            } else {
                j += 1;
            }
        }
        // `i` is already incremented to the next position of a box that wasn't removed.
    }
    selected_boxes
}

// 4. Output Parsing Function (Basic Implementation)
pub fn parse_yolo_output<B: Backend>(
    output_tensor: Tensor<B, 4>, // Shape: [batch_size, num_anchors * (5 + num_classes), grid_h, grid_w]
    anchors: &[(f32, f32)],      // Anchors for this specific scale, e.g., [(10.,13.), (16.,30.), (33.,23.)]
    num_classes: usize,
    confidence_threshold: f32,
    input_dim: (u32, u32), // (width, height) of the input image to the network
) -> Vec<BoundingBox> {
    let mut bboxes: Vec<BoundingBox> = Vec::new();

    let (batch_size, _, grid_h, grid_w) = output_tensor.dims();
    if batch_size != 1 {
        // For simplicity, this example parser assumes batch_size = 1
        // Production code should handle batch_size > 1
        eprintln!("Warning: parse_yolo_output currently only supports batch_size = 1 for simplicity.");
        // return bboxes; // Or process only the first image in batch
    }

    let num_anchors = anchors.len();
    // Expected output_tensor.dims()[1] == num_anchors * (5 + num_classes)
    
    // Convert tensor to Data for easier CPU access.
    // This is a blocking call and might be slow. For performance, keep operations on tensor if possible.
    let output_data: Data<B::FloatElem, 4> = output_tensor.into_data();

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            for anchor_idx in 0..num_anchors {
                let data_idx_base = anchor_idx * (5 + num_classes);

                // Assuming output_data is flat for the channel dimension or use output_data.get([0, channel_idx, gy, gx])
                // For simplicity, let's assume we can index it directly or have a helper to get values.
                // Burn's Data struct doesn't directly support [b, c, h, w] indexing like a direct array.
                // We need to calculate the flat index or use slice + reshape if we want to access it like that.
                // Let's simulate accessing the data sequentially for the channel dimension.
                // The actual data layout in `Data.value` is a flat Vec<B::FloatElem>.
                // Index = b * (num_channels * grid_h * grid_w) + c * (grid_h * grid_w) + gy * grid_w + gx
                
                let tx_val_flat_idx = 0 * (num_anchors * (5 + num_classes) * grid_h * grid_w) + // batch 0
                                      (data_idx_base + 0) * (grid_h * grid_w) +
                                      gy * grid_w + gx;
                let ty_val_flat_idx = tx_val_flat_idx + (grid_h * grid_w); // Next element in channel for same (gy,gx)
                let tw_val_flat_idx = ty_val_flat_idx + (grid_h * grid_w);
                let th_val_flat_idx = tw_val_flat_idx + (grid_h * grid_w);
                let conf_val_flat_idx = th_val_flat_idx + (grid_h * grid_w);

                let tx: f32 = output_data.value[tx_val_flat_idx as usize].elem();
                let ty: f32 = output_data.value[ty_val_flat_idx as usize].elem();
                let tw: f32 = output_data.value[tw_val_flat_idx as usize].elem();
                let th: f32 = output_data.value[th_val_flat_idx as usize].elem();
                let objectness: f32 = output_data.value[conf_val_flat_idx as usize].elem();

                let confidence = 1.0 / (1.0 + (-objectness).exp()); // Manual sigmoid

                if confidence < confidence_threshold {
                    continue;
                }

                let (anchor_w, anchor_h) = anchors[anchor_idx];

                // Decode box coordinates
                let bx = (1.0 / (1.0 + (-tx).exp())) + gx as f32; // sigmoid(tx) + cx
                let by = (1.0 / (1.0 + (-ty).exp())) + gy as f32; // sigmoid(ty) + cy
                let bw = anchor_w * tw.exp(); // anchor_w * exp(tw)
                let bh = anchor_h * th.exp(); // anchor_h * exp(th)

                // Normalize grid cell coordinates to input image dimensions
                let stride_x = input_dim.0 as f32 / grid_w as f32;
                let stride_y = input_dim.1 as f32 / grid_h as f32;

                let final_x = (bx - bw / 2.0) * stride_x; // Center x to top-left x, then scale
                let final_y = (by - bh / 2.0) * stride_y; // Center y to top-left y, then scale
                let final_w = bw * stride_x;
                let final_h = bh * stride_y;

                // Class probabilities
                let mut class_id = 0;
                let mut max_class_prob = 0.0_f32;

                if num_classes > 0 {
                    let mut class_scores = Vec::with_capacity(num_classes);
                    for i in 0..num_classes {
                        let class_score_flat_idx = conf_val_flat_idx + ((i + 1) * grid_h * grid_w);
                        class_scores.push(output_data.value[class_score_flat_idx as usize].elem());
                    }
                    
                    // Softmax class_scores (manual or use burn::tensor::activation::softmax if on tensor)
                    let mut exp_sum = 0.0;
                    for score in &class_scores {
                        exp_sum += score.exp();
                    }
                    let mut softmaxed_scores = Vec::new();
                    for score in &class_scores {
                        softmaxed_scores.push(score.exp() / exp_sum);
                    }

                    for (i, &prob) in softmaxed_scores.iter().enumerate() {
                        if prob > max_class_prob {
                            max_class_prob = prob;
                            class_id = i;
                        }
                    }
                }
                // If num_classes is 0, class_id remains 0, and confidence is just objectness_confidence.
                // For object detection, you usually have classes.
                // The final confidence can be objectness_confidence * max_class_prob.
                let final_confidence = confidence * max_class_prob; // Or just `confidence` if no classes or if that's preferred

                if final_confidence >= confidence_threshold {
                     bboxes.push(BoundingBox {
                        x: final_x,
                        y: final_y,
                        width: final_w,
                        height: final_h,
                        class_id,
                        confidence: final_confidence,
                    });
                }
            }
        }
    }
    bboxes
}


// 5. Weight Loading Function (Conceptual Placeholder)
pub fn load_darknet_weights<B: Backend>(
    model: YoloV3<B>, // Take by value and return, or take mut ref
    filepath: &str,
) -> Result<YoloV3<B>, Box<dyn std::error::Error>> {
    println!("Placeholder: Attempting to load Darknet weights from {}", filepath);
    println!("Actual weight loading is not implemented in this example.");
    // In a real scenario, this would involve:
    // 1. Reading the .weights file (custom binary format).
    // 2. Iterating through model layers and loading weights sequentially.
    // 3. Handling potential mismatches in layer types or shapes.
    // 4. Burn models store weights in Tensors (e.g., conv.weight, bn.weight, bn.bias).
    //    You'd need to create Data<f32, D> and then Tensor::from_data(data).to_device(&device)
    //    and assign it to the model's parameters. This is tricky because model parameters are
    //    typically private and updated via `load_record`. A more Burn-idiomatic way would be
    //    to parse weights into a Burn model record and then use `model.load_record(record)`.
    Ok(model) // Return the model as is, or a new one if it was mutated.
}
