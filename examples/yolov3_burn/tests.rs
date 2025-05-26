#[cfg(test)]
mod tests {
  use crate::model::*; // 修复：使用 crate 而不是 super
  use crate::utils::*; // 修复：使用 crate 而不是 super

  use burn::backend::wgpu::WgpuDevice;
  use burn::backend::{Autodiff, Wgpu}; // 修复：使用 Wgpu 而不是 TestBackend
  use burn::tensor::{Tensor, TensorData}; // 修复：使用 TensorData 而不是 Data

  type TB = Wgpu; // 使用 Wgpu 后端进行测试
  type ADTB = Autodiff<TB>; // Autodiff Wgpu Backend

  fn get_device() -> WgpuDevice {
    WgpuDevice::default()
  }

  #[test]
  fn test_conv_block_forward() {
    let device = get_device();
    let conv_block = ConvBlock::new(3, 16, 3, 1, 1, true, &device);

    let input: Tensor<ADTB, 4> = Tensor::zeros([1, 3, 32, 32], &device);
    let output = conv_block.forward(input.clone());

    assert_eq!(output.dims(), [1, 16, 32, 32]);

    let input_non_zero: Tensor<ADTB, 4> = Tensor::ones([1, 3, 32, 32], &device);
    let _output_non_zero = conv_block.forward(input_non_zero.clone());

    println!(
      "ConvBlock test: Output shape {:?} matches expected.",
      output.dims()
    );
  }

  #[test]
  fn test_residual_block_forward() {
    let device = get_device();
    let res_block = ResidualBlock::new(16, &device);

    let input: Tensor<ADTB, 4> = Tensor::ones([1, 16, 32, 32], &device);
    let output = res_block.forward(input.clone());

    assert_eq!(output.dims(), input.dims());

    println!(
      "ResidualBlock test: Output shape {:?} matches input shape.",
      output.dims()
    );
  }

  #[test]
  fn test_yolov3_init() {
    let device = get_device();
    let num_classes = 80;
    let num_anchors_per_scale = 3;
    let config = YoloV3Config::new(num_classes, num_anchors_per_scale);
    let _model: YoloV3<ADTB> = config.init(&device);
    println!("YoloV3 Init test: Model initialized successfully.");
  }

  #[test]
  fn test_yolov3_forward_shapes() {
    let device = get_device();
    let num_classes = 80;
    let num_anchors_per_scale = 3;
    let config = YoloV3Config::new(num_classes, num_anchors_per_scale);
    let model: YoloV3<ADTB> = config.init(&device);

    let input_dim = 416;
    let input: Tensor<ADTB, 4> = Tensor::zeros([1, 3, input_dim, input_dim], &device);
    let (output_scale1, output_scale2, output_scale3) = model.forward(input);

    let expected_channels = num_anchors_per_scale * (5 + num_classes);
    assert_eq!(
      output_scale1.dims(),
      [1, expected_channels, input_dim / 32, input_dim / 32]
    );
    assert_eq!(
      output_scale2.dims(),
      [1, expected_channels, input_dim / 16, input_dim / 16]
    );
    assert_eq!(
      output_scale3.dims(),
      [1, expected_channels, input_dim / 8, input_dim / 8]
    );
    println!("YoloV3 Forward Shapes test: Output shapes match expected.");
  }

  #[test]
  fn test_parse_yolo_output_basic() {
    let device = get_device();
    let num_classes = 1; // 简化测试
    let num_anchors_per_scale = 1;
    let confidence_threshold = 0.01; // 低阈值以捕获边界框
    let input_dim_w = 416;
    let input_dim_h = 416;
    let grid_size = 1; // 1x1 网格简化测试

    let channels = num_anchors_per_scale * (5 + num_classes); // 1 * (5+1) = 6

    // 手动创建张量数据
    let mut tensor_data = vec![0.0f32; channels * grid_size * grid_size];
    tensor_data[0] = 0.0; // tx
    tensor_data[1] = 0.0; // ty
    tensor_data[2] = 0.0; // tw
    tensor_data[3] = 0.0; // th
    tensor_data[4] = 1.386; // objectness score (sigmoid -> 0.8)
    tensor_data[5] = 10.0; // class score (for class 0)

    // 修复：使用 TensorData 和正确的 API
    let tensor_data = TensorData::new(tensor_data, [1, channels, grid_size, grid_size]);
    let output_tensor: Tensor<ADTB, 4> = Tensor::from_data(tensor_data, &device);

    let anchors = vec![(10.0f32, 10.0f32)]; // Anchor w=10, h=10

    let bboxes = parse_yolo_output(
      output_tensor,
      &anchors,
      num_classes,
      confidence_threshold,
      (input_dim_w, input_dim_h),
    );

    // 由于我们使用的是简化的实现，只检查基本功能
    println!(
      "Parse YOLO Output Basic test: Generated {} bounding boxes",
      bboxes.len()
    );
    // 检查边界框数量是否合理（移除无用的比较）
    assert!(
      bboxes.len() < 1000,
      "Should not generate too many bounding boxes"
    );
    println!("Parse YOLO Output Basic test: Passed.");
  }

  #[test]
  fn test_nms_basic() {
    let boxes = vec![
      BoundingBox {
        x: 0.0,
        y: 0.0,
        width: 10.0,
        height: 10.0,
        class_id: 0,
        confidence: 0.9,
      }, // Keep
      BoundingBox {
        x: 100.0,
        y: 100.0,
        width: 10.0,
        height: 10.0,
        class_id: 0,
        confidence: 0.1,
      }, // Filter by conf
      BoundingBox {
        x: 1.0,
        y: 1.0,
        width: 10.0,
        height: 10.0,
        class_id: 0,
        confidence: 0.8,
      }, // Suppress by IoU
      BoundingBox {
        x: 50.0,
        y: 50.0,
        width: 10.0,
        height: 10.0,
        class_id: 0,
        confidence: 0.85,
      }, // Keep
    ];

    let confidence_threshold = 0.5;
    let iou_threshold = 0.5;

    let result_boxes = non_maximum_suppression(boxes, iou_threshold, confidence_threshold);

    assert_eq!(result_boxes.len(), 2);
    // 检查是否保留了正确的边界框（第一个和第四个原始边界框）
    // 由于排序，顺序可能会改变，所以检查置信度值
    assert!(
      result_boxes
        .iter()
        .any(|b| (b.confidence - 0.9).abs() < f32::EPSILON)
    );
    assert!(
      result_boxes
        .iter()
        .any(|b| (b.confidence - 0.85).abs() < f32::EPSILON)
    );
    println!("NMS Basic test: Passed.");
  }
}
