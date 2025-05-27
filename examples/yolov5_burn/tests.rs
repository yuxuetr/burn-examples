#[cfg(test)]
mod tests {
  use crate::model::*; // Import YOLOv5 model components
  use crate::utils::*; // Import utility functions like NMS, parsing, anchors

  use burn::backend::wgpu::WgpuDevice;
  use burn::backend::{Autodiff, Wgpu};
  use burn::tensor::{Tensor, Data, Shape, Float}; // Updated for modern Burn
  use burn::prelude::*;


  type TB = Wgpu; 
  type ADTB = Autodiff<TB>;

  fn get_device() -> WgpuDevice {
    // WgpuDevice::default() // This might pick integrated GPU. Be specific if needed.
    // Forcing discrete GPU if available, otherwise fallback.
    // You might need to adjust this based on your environment or use WGPU_POWER_PREF env var.
    WgpuDevice::BestAvailable
  }

  #[test]
  fn test_conv_block_forward() {
    let device = get_device();
    let conv_block = ConvBlock::<ADTB>::new(3, 16, 3, 1, 1, true, &device);

    let input: Tensor<ADTB, 4> = Tensor::zeros(Shape::new([1, 3, 32, 32]), &device);
    let output = conv_block.forward(input.clone());

    assert_eq!(output.dims(), [1, 16, 32, 32]);
    println!("ConvBlock test: Output shape {:?} matches expected.", output.dims());
  }

  #[test]
  fn test_bottleneck_forward() {
    let device = get_device();
    // Test with residual connection
    let bottleneck_res = Bottleneck::<ADTB>::new(16, 16, true, 0.5, &device);
    let input_res: Tensor<ADTB, 4> = Tensor::ones(Shape::new([1, 16, 32, 32]), &device);
    let output_res = bottleneck_res.forward(input_res.clone());
    assert_eq!(output_res.dims(), input_res.dims());
    println!("Bottleneck (residual) test: Output shape {:?} matches input shape.", output_res.dims());

    // Test without residual connection (channels differ)
    let bottleneck_no_res = Bottleneck::<ADTB>::new(16, 32, false, 0.5, &device);
    let input_no_res: Tensor<ADTB, 4> = Tensor::ones(Shape::new([1, 16, 32, 32]), &device);
    let output_no_res = bottleneck_no_res.forward(input_no_res.clone());
    assert_eq!(output_no_res.dims(), [1, 32, 32, 32]);
     println!("Bottleneck (no residual) test: Output shape {:?} as expected.", output_no_res.dims());
  }

  #[test]
  fn test_csp_stage_forward() {
    let device = get_device();
    let csp_stage = CSPStage::<ADTB>::new(32, 64, 3, true, 0.5, &device); // downsample = true
    let input: Tensor<ADTB, 4> = Tensor::ones(Shape::new([1, 32, 64, 64]), &device);
    let output = csp_stage.forward(input.clone());
    // Expected output: [batch, out_channels, H/2, W/2] due to downsampling
    assert_eq!(output.dims(), [1, 64, 32, 32]);
    println!("CSPStage test: Output shape {:?} as expected.", output.dims());
  }
  
  #[test]
  fn test_sppf_forward() {
    let device = get_device();
    let sppf = SPPF::<ADTB>::new(64, 64, 5, &device);
    let input: Tensor<ADTB, 4> = Tensor::ones(Shape::new([1, 64, 32, 32]), &device);
    let output = sppf.forward(input.clone());
    assert_eq!(output.dims(), [1, 64, 32, 32]); // SPPF should maintain H, W, and output channels
    println!("SPPF test: Output shape {:?} as expected.", output.dims());
  }


  #[test]
  fn test_yolov5_init() {
    let device = get_device();
    let num_classes = 80;
    let num_anchors_per_scale = 3;
    // Use the yolov5s configuration
    let config = YoloV5Config::yolov5s(num_classes, num_anchors_per_scale);
    let _model: YoloV5<ADTB> = config.init(&device);
    println!("YoloV5 Init test: Model initialized successfully using yolov5s config.");
  }

  #[test]
  fn test_yolov5_forward_shapes() {
    let device = get_device();
    let num_classes = 80;
    let num_anchors_per_scale = 3; // Standard for YOLOv5
    let config = YoloV5Config::yolov5s(num_classes, num_anchors_per_scale);
    let model: YoloV5<ADTB> = config.init(&device);

    let input_dim = 640; // Common input size for YOLOv5
    let input: Tensor<ADTB, 4> = Tensor::zeros(Shape::new([1, 3, input_dim, input_dim]), &device);
    let (output_p3, output_p4, output_p5) = model.forward(input);

    // Expected output shape: [batch_size, num_anchors, grid_h, grid_w, (5 + num_classes)]
    let expected_attributes = 5 + num_classes;

    // P3 output (stride 8)
    let expected_grid_p3 = input_dim / 8;
    assert_eq!(
      output_p3.dims(),
      [1, num_anchors_per_scale, expected_grid_p3, expected_grid_p3, expected_attributes]
    );

    // P4 output (stride 16)
    let expected_grid_p4 = input_dim / 16;
    assert_eq!(
      output_p4.dims(),
      [1, num_anchors_per_scale, expected_grid_p4, expected_grid_p4, expected_attributes]
    );

    // P5 output (stride 32)
    let expected_grid_p5 = input_dim / 32;
    assert_eq!(
      output_p5.dims(),
      [1, num_anchors_per_scale, expected_grid_p5, expected_grid_p5, expected_attributes]
    );
    println!("YoloV5 Forward Shapes test: Output shapes match expected for P3, P4, P5.");
  }

  #[test]
  fn test_parse_yolov5_output_basic() {
    let device = get_device();
    let num_classes = 1; // Simplified for testing
    let num_anchors_per_scale = 3; // Must match model output
    let confidence_threshold = 0.01; 
    let network_input_dim_w = 640;
    let network_input_dim_h = 640;
    let original_img_dims = (640u32, 480u32); // Example original image size

    let attributes = 5 + num_classes;

    // Create a dummy output tensor for one scale (e.g., P3, stride 8)
    let grid_size = network_input_dim_w / 8; // 80x80 for 640 input
    let stride = 8.0f32;
    let anchors_p3 = get_yolov5s_anchors()[0].clone(); // Anchors for P3

    // Create some data: [batch, num_anchors, grid_h, grid_w, attributes]
    // Let's make one bounding box "active"
    // (tx, ty, tw, th, conf, class_probs...)
    // For sigmoid(tx) = 0.5 (center of cell), tx = 0
    // For sigmoid(tw) = 0.5 (anchor size), tw = 0
    // For sigmoid(conf) = 0.8, conf = logit(0.8) approx 1.386
    // For sigmoid(class_prob) = 0.9, class_prob = logit(0.9) approx 2.197
    let mut tensor_values = vec![0.0f32; 1 * num_anchors_per_scale * grid_size * grid_size * attributes];
    
    // Activate one anchor in one cell (e.g., first anchor, cell 10,10)
    let target_anchor_idx = 0;
    let target_gx = 10;
    let target_gy = 10;
    let base_idx = target_anchor_idx * (grid_size * grid_size * attributes) +
                   target_gy * (grid_size * attributes) +
                   target_gx * attributes;

    tensor_values[base_idx + 0] = 0.0; // tx (sigmoid -> 0.5)
    tensor_values[base_idx + 1] = 0.0; // ty (sigmoid -> 0.5)
    tensor_values[base_idx + 2] = 0.0; // tw (sigmoid -> 0.5, so (2*0.5)^2 = 1 * anchor_w)
    tensor_values[base_idx + 3] = 0.0; // th (sigmoid -> 0.5, so (2*0.5)^2 = 1 * anchor_h)
    tensor_values[base_idx + 4] = 1.386; // conf (sigmoid -> 0.8)
    tensor_values[base_idx + 5] = 2.197; // class 0 prob (sigmoid -> 0.9)

    let output_shape = Shape::new([1, num_anchors_per_scale, grid_size, grid_size, attributes]);
    let tensor_data = Data::<Float, 5>::new(tensor_values, output_shape);
    let output_tensor: Tensor<ADTB, 5> = Tensor::from_data(tensor_data.convert(), &device);

    let bboxes = parse_yolo_output(
      output_tensor,
      &anchors_p3,
      num_classes,
      confidence_threshold,
      stride,
      original_img_dims,
      (network_input_dim_w, network_input_dim_h),
    );
    
    println!("Parse YOLOv5 Output Basic test: Generated {} bounding boxes", bboxes.len());
    assert_eq!(bboxes.len(), 1, "Should detect one bounding box based on dummy data");

    if let Some(bbox) = bboxes.first() {
        println!("Detected BBox: {:?}", bbox);
        // Check if coordinates are reasonable (they should be scaled by stride and anchors)
        // Expected center x roughly: (0.5 + 10) * 8 = 10.5 * 8 = 84 (before letterbox adjustment)
        // Expected width roughly: (2*0.5)^2 * anchor_w = anchors_p3[0].0 (before letterbox adjustment)
        assert!(bbox.confidence > 0.7, "Confidence should be high"); // 0.8 * 0.9 = 0.72
        assert_eq!(bbox.class_id, 0, "Class ID should be 0");
    }
    println!("Parse YOLOv5 Output Basic test: Passed.");
  }
  
  #[test]
  fn test_parse_all_yolov5_outputs() {
    let device = get_device();
    let num_classes = 2;
    let num_anchors_per_scale = 3;
    let confidence_threshold = 0.1;
    let network_dims = (640u32, 640u32);
    let original_img_dims = (1280u32, 720u32);
    let all_yolov5_anchors = get_yolov5s_anchors();
    let strides = [8.0f32, 16.0f32, 32.0f32];
    let attributes = 5 + num_classes;

    let make_dummy_output = |grid_size_div: u32, anchor_idx_to_activate: usize, gx:usize, gy:usize| -> Tensor<ADTB, 5> {
        let grid_size = network_dims.0 / grid_size_div;
        let mut tensor_values = vec![0.0f32; 1 * num_anchors_per_scale * grid_size * grid_size * attributes];
        if gx < grid_size && gy < grid_size && anchor_idx_to_activate < num_anchors_per_scale {
            let base_idx = anchor_idx_to_activate * (grid_size * grid_size * attributes) +
                           gy * (grid_size * attributes) +
                           gx * attributes;
            tensor_values[base_idx + 0] = 0.0; 
            tensor_values[base_idx + 1] = 0.0; 
            tensor_values[base_idx + 2] = 0.0; 
            tensor_values[base_idx + 3] = 0.0; 
            tensor_values[base_idx + 4] = 1.386; // conf ~0.8
            tensor_values[base_idx + 5 + 0] = 2.197; // class 0 prob ~0.9
            tensor_values[base_idx + 5 + 1] = -2.197; // class 1 prob ~0.1
        }
        let output_shape = Shape::new([1, num_anchors_per_scale, grid_size, grid_size, attributes]);
        let tensor_data = Data::<Float, 5>::new(tensor_values, output_shape);
        Tensor::from_data(tensor_data.convert(), &device)
    };

    let output_p3 = make_dummy_output(8, 0, 10, 10); // Activate one box in P3
    let output_p4 = make_dummy_output(16, 1, 5, 5);  // Activate one box in P4
    let output_p5 = make_dummy_output(32, 2, 2, 2);  // Activate one box in P5
    let no_box_output: Tensor<ADTB, 5> = Tensor::zeros_like(&output_p3);


    let bboxes1 = parse_all_yolo_outputs(
        (output_p3.clone(), output_p4.clone(), output_p5.clone()),
        &all_yolov5_anchors,
        num_classes,
        confidence_threshold,
        &strides,
        original_img_dims,
        network_dims,
    );
    assert_eq!(bboxes1.len(), 3, "Should detect three boxes from three scales");

    let bboxes2 = parse_all_yolo_outputs(
        (output_p3.clone(), no_box_output.clone(), no_box_output.clone()),
        &all_yolov5_anchors,
        num_classes,
        confidence_threshold,
        &strides,
        original_img_dims,
        network_dims,
    );
    assert_eq!(bboxes2.len(), 1, "Should detect one box from P3 only");
    println!("Parse All YOLOv5 Outputs test: Passed.");
  }


  #[test]
  fn test_nms_basic() {
    // Note: BoundingBox x,y are center coordinates now
    let boxes = vec![
      BoundingBox { x_center: 5.0, y_center: 5.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.9 }, // Keep
      BoundingBox { x_center: 105.0, y_center: 105.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.85 }, // Keep
      BoundingBox { x_center: 6.0, y_center: 6.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.8 }, // Suppress by IoU with first box
    ];

    // NMS does not filter by confidence, that's done by parse_yolo_output
    // However, typical use of NMS is on boxes *after* confidence filtering.
    // For this standalone test, the input boxes are assumed to be above threshold.
    let iou_threshold = 0.5; 
    
    let result_boxes = non_maximum_suppression(boxes, iou_threshold, 0.0); // conf_threshold = 0 for this test as it's pre-filtered

    assert_eq!(result_boxes.len(), 2);
    // Check confidences of kept boxes
    assert!(result_boxes.iter().any(|b| (b.confidence - 0.9).abs() < f32::EPSILON));
    assert!(result_boxes.iter().any(|b| (b.confidence - 0.85).abs() < f32::EPSILON));
    println!("NMS Basic test: Passed with updated BoundingBox.");
  }
}
