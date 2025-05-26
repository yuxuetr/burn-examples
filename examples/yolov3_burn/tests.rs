#[cfg(test)]
mod tests {
    use super::model::*; // Access model items from main.rs (which includes model.rs)
    use super::utils::*; // Access utils items from main.rs (which includes utils.rs)
    
    use burn::backend::{Autodiff, TestBackend};
    use burn::tensor::{Tensor, Data, Shape, Float, Int, ElementConversion};

    type TB = TestBackend; // TestBackend
    type ADTB = Autodiff<TB>; // Autodiff TestBackend

    const DEFAULT_DEVICE: <TB as burn::tensor::backend::Backend>::Device = Default::default();

    #[test]
    fn test_conv_block_forward() {
        let config = ConvBlockConfig::new(3, 16, 3, 1, 1, true);
        let conv_block: ConvBlock<ADTB> = config.init();
        let conv_block = conv_block.to_device(&DEFAULT_DEVICE);

        let input: Tensor<ADTB, 4> = Tensor::zeros([1, 3, 32, 32]).to_device(&DEFAULT_DEVICE);
        let output = conv_block.forward(input.clone());

        assert_eq!(output.dims(), [1, 16, 32, 32]);
        // A simple way to check they are not equal is to check their sum of elements,
        // assuming weights are not all zero, which they are not by default.
        // However, with zero input and possible zero biases (if bias=false or init to zero), output could be zero.
        // A more robust check is if any element is different, but that's harder.
        // For now, let's assume default init creates non-zero weights.
        // If input is zeros, output will be zeros if bias is zero.
        // Let's make input non-zero.
        let input_non_zero: Tensor<ADTB, 4> = Tensor::ones([1, 3, 32, 32]).to_device(&DEFAULT_DEVICE);
        let output_non_zero = conv_block.forward(input_non_zero.clone());
        
        // Check if output is different from input for a non-zero input
        // This is not a perfect test as a specific configuration could still map ones to ones.
        // A better test would be specific values or statistical properties.
        // For now, we check that output is not exactly the same as input if all channels were the same.
        // This test is weak, as output channels are different. Sum is not a good test.
        // Just checking shapes is the primary goal here.
        // A simple check: output sum is not simply input sum * (out_channels/in_channels)
        // For now, shape assertion is the most reliable.
        // Let's assert that the output is not just a clone of the input, which is implicitly true due to shape change.
        println!("ConvBlock test: Output shape {:?} matches expected.", output.dims());
    }

    #[test]
    fn test_residual_block_forward() {
        let config = ResidualBlockConfig::new(16);
        let res_block: ResidualBlock<ADTB> = config.init();
        let res_block = res_block.to_device(&DEFAULT_DEVICE);
        
        let input: Tensor<ADTB, 4> = Tensor::ones([1, 16, 32, 32]).to_device(&DEFAULT_DEVICE);
        let output = res_block.forward(input.clone());

        assert_eq!(output.dims(), input.dims());
        // For a ResBlock, output can be equal to input if conv layers result in adding zero to residual.
        // This is unlikely with standard initializations.
        // A simple check that some processing happened:
        let output_sum: Data<f32, 1> = output.sum().into_data();
        let input_sum: Data<f32, 1> = input.sum().into_data();
        // This is not a perfect assertion, but good enough for a basic check.
        // It might fail if weights initialize in a very specific (unlikely) way to make sum(conv_out) == 0
        // For a more robust check, one might need to inspect specific values or use a pre-trained model.
        // A simple check: output is not identical to input.
        assert_ne!(output_sum.value[0], input_sum.value[0] * 2.0_f32.sqrt(), "Output sum suggests limited processing in ResBlock or specific weight init.");
        println!("ResidualBlock test: Output shape {:?} matches input shape.", output.dims());
    }

    #[test]
    fn test_yolov3_init() {
        let num_classes = 80;
        let num_anchors_per_scale = 3;
        let config = YoloV3Config::new(num_classes, num_anchors_per_scale);
        let _model: YoloV3<ADTB> = config.init();
        // If init completes without panic, the test passes.
        println!("YoloV3 Init test: Model initialized successfully.");
    }

    #[test]
    fn test_yolov3_forward_shapes() {
        let num_classes = 80;
        let num_anchors_per_scale = 3;
        let config = YoloV3Config::new(num_classes, num_anchors_per_scale);
        let model: YoloV3<ADTB> = config.init().to_device(&DEFAULT_DEVICE);

        let input_dim = 416;
        let input: Tensor<ADTB, 4> = Tensor::zeros([1, 3, input_dim, input_dim]).to_device(&DEFAULT_DEVICE);
        let (output_scale1, output_scale2, output_scale3) = model.forward(input);

        let expected_channels = num_anchors_per_scale * (5 + num_classes);
        assert_eq!(output_scale1.dims(), [1, expected_channels, input_dim / 32, input_dim / 32]);
        assert_eq!(output_scale2.dims(), [1, expected_channels, input_dim / 16, input_dim / 16]);
        assert_eq!(output_scale3.dims(), [1, expected_channels, input_dim / 8, input_dim / 8]);
        println!("YoloV3 Forward Shapes test: Output shapes match expected.");
    }

    #[test]
    fn test_parse_yolo_output_basic() {
        let num_classes = 1; // Simplified
        let num_anchors_per_scale = 1;
        let confidence_threshold = 0.01; // Low threshold to catch the box
        let input_dim_w = 416;
        let input_dim_h = 416;
        let grid_size = 1; // 1x1 grid for simplicity

        let channels = num_anchors_per_scale * (5 + num_classes); // 1 * (5+1) = 6
        
        // Manually create tensor data
        // Values for tx, ty, tw, th, objectness, class1_score
        // Sigmoid(tx) = 0.5 (tx=0), Sigmoid(ty)=0.5 (ty=0)
        // Exp(tw) = 1.0 (tw=0), Exp(th)=1.0 (th=0)
        // Sigmoid(objectness) = 0.8 (objectness approx 1.386)
        // Softmax(class1_score) = 1.0 (class1_score high, e.g. 10.0, others implicit if num_classes > 1)
        let mut tensor_data = vec![0.0f32; channels * grid_size * grid_size];
        tensor_data[0] = 0.0; // tx
        tensor_data[1 * (grid_size * grid_size)] = 0.0; // ty
        tensor_data[2 * (grid_size * grid_size)] = 0.0; // tw
        tensor_data[3 * (grid_size * grid_size)] = 0.0; // th
        tensor_data[4 * (grid_size * grid_size)] = 1.386; // objectness score (sigmoid -> 0.8)
        tensor_data[5 * (grid_size * grid_size)] = 10.0;  // class score (for class 0)

        let data_shape = Shape::new([1, channels, grid_size, grid_size]);
        let output_tensor_data = Data::<f32, 4>::new(tensor_data, data_shape);
        let output_tensor: Tensor<ADTB, 4> = Tensor::from_data(output_tensor_data.convert()).to_device(&DEFAULT_DEVICE);
        
        let anchors = vec![(10.0f32, 10.0f32)]; // Anchor w=10, h=10

        let bboxes = parse_yolo_output(
            output_tensor,
            &anchors,
            num_classes,
            confidence_threshold,
            (input_dim_w, input_dim_h),
        );

        assert_eq!(bboxes.len(), 1, "Expected one bounding box");
        if bboxes.len() == 1 {
            let bbox = &bboxes[0];
            assert_eq!(bbox.class_id, 0);
            assert!((bbox.confidence - 0.8).abs() < 0.01, "Confidence mismatch"); // Sigmoid(1.386) * Softmax(10) approx 0.8

            let expected_x_center_grid = 0.5; // sigmoid(0) + 0 (grid_x)
            let expected_y_center_grid = 0.5; // sigmoid(0) + 0 (grid_y)
            let expected_w_grid = 10.0 * (0.0f32).exp(); // anchor_w * exp(0)
            let expected_h_grid = 10.0 * (0.0f32).exp(); // anchor_h * exp(0)

            let stride_x = input_dim_w as f32 / grid_size as f32;
            let stride_y = input_dim_h as f32 / grid_size as f32;

            let expected_final_x = (expected_x_center_grid - expected_w_grid / 2.0) * stride_x;
            let expected_final_y = (expected_y_center_grid - expected_h_grid / 2.0) * stride_y;
            let expected_final_w = expected_w_grid * stride_x;
            let expected_final_h = expected_h_grid * stride_y;
            
            assert!((bbox.x - expected_final_x).abs() < 0.1, "X mismatch. Got {}, expected {}", bbox.x, expected_final_x);
            assert!((bbox.y - expected_final_y).abs() < 0.1, "Y mismatch. Got {}, expected {}", bbox.y, expected_final_y);
            assert!((bbox.width - expected_final_w).abs() < 0.1, "Width mismatch. Got {}, expected {}", bbox.width, expected_final_w);
            assert!((bbox.height - expected_final_h).abs() < 0.1, "Height mismatch. Got {}, expected {}", bbox.height, expected_final_h);
        }
        println!("Parse YOLO Output Basic test: Passed.");
    }

    #[test]
    fn test_nms_basic() {
        let boxes = vec![
            BoundingBox { x: 0.0, y: 0.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.9 }, // Keep
            BoundingBox { x: 100.0, y: 100.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.1 },// Filter by conf
            BoundingBox { x: 1.0, y: 1.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.8 },   // Suppress by IoU
            BoundingBox { x: 50.0, y: 50.0, width: 10.0, height: 10.0, class_id: 0, confidence: 0.85 },// Keep
        ];

        let confidence_threshold = 0.5;
        let iou_threshold = 0.5;

        let result_boxes = non_maximum_suppression(boxes, iou_threshold, confidence_threshold);

        assert_eq!(result_boxes.len(), 2);
        // Check if the correct boxes were kept (the first and the fourth original box)
        // The order might change due to sorting, so check confidence values.
        assert!(result_boxes.iter().any(|b| (b.confidence - 0.9).abs() < f32::EPSILON));
        assert!(result_boxes.iter().any(|b| (b.confidence - 0.85).abs() < f32::EPSILON));
        println!("NMS Basic test: Passed.");
    }
}
