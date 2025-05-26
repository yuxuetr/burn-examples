use burn::{
  module::Module,
  nn::{
    BatchNorm, BatchNormConfig, LeakyRelu, LeakyReluConfig,
    conv::{Conv2d, Conv2dConfig},
  },
  tensor::{Device, Tensor, backend::Backend},
};

// ConvBlock Definition
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
  conv: Conv2d<B>,
  bn: BatchNorm<B, 2>,
  activation: LeakyRelu,
}

impl<B: Backend> ConvBlock<B> {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
    device: &Device<B>,
  ) -> Self {
    Self {
      conv: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(burn::nn::PaddingConfig2d::Explicit(padding, padding))
        .with_bias(bias)
        .init(device),
      bn: BatchNormConfig::new(out_channels).init(device),
      activation: LeakyReluConfig::new().with_negative_slope(0.1).init(),
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv.forward(input);
    let x = self.bn.forward(x);
    self.activation.forward(x)
  }
}

// ResidualBlock Definition
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
  conv1: ConvBlock<B>,
  conv2: ConvBlock<B>,
}

impl<B: Backend> ResidualBlock<B> {
  pub fn new(channels: usize, device: &Device<B>) -> Self {
    let half_channels = channels / 2;
    Self {
      conv1: ConvBlock::new(channels, half_channels, 1, 1, 0, true, device),
      conv2: ConvBlock::new(half_channels, channels, 3, 1, 1, true, device),
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let residual = input.clone();
    let x = self.conv1.forward(input);
    let x = self.conv2.forward(x);
    x + residual
  }
}

// YoloV3 Model Definition
#[derive(Module, Debug)]
pub struct YoloV3<B: Backend> {
  // Darknet-53 Backbone
  conv_initial: ConvBlock<B>,
  downsample1: ConvBlock<B>,
  res_block_1_1: ResidualBlock<B>,

  downsample2: ConvBlock<B>,
  res_block_2_1: ResidualBlock<B>,
  res_block_2_2: ResidualBlock<B>,

  downsample3: ConvBlock<B>,
  res_blocks_3: Vec<ResidualBlock<B>>,

  downsample4: ConvBlock<B>,
  res_blocks_4: Vec<ResidualBlock<B>>,

  downsample5: ConvBlock<B>,
  res_blocks_5: Vec<ResidualBlock<B>>,

  // YOLO Head
  head_convs_scale1: Vec<ConvBlock<B>>,
  detect_layer_scale1: Conv2d<B>,

  head_convs_scale2: Vec<ConvBlock<B>>,
  detect_layer_scale2: Conv2d<B>,

  head_convs_scale3: Vec<ConvBlock<B>>,
  detect_layer_scale3: Conv2d<B>,
}

pub struct YoloV3Config {
  pub num_classes: usize,
  pub num_anchors_per_scale: usize,
}

impl YoloV3Config {
  pub fn new(num_classes: usize, num_anchors_per_scale: usize) -> Self {
    Self {
      num_classes,
      num_anchors_per_scale,
    }
  }

  pub fn init<B: Backend>(&self, device: &Device<B>) -> YoloV3<B> {
    let bias = true;

    // Darknet-53 Backbone
    let conv_initial = ConvBlock::new(3, 32, 3, 1, 1, bias, device);
    let downsample1 = ConvBlock::new(32, 64, 3, 2, 1, bias, device);
    let res_block_1_1 = ResidualBlock::new(64, device);

    let downsample2 = ConvBlock::new(64, 128, 3, 2, 1, bias, device);
    let res_block_2_1 = ResidualBlock::new(128, device);
    let res_block_2_2 = ResidualBlock::new(128, device);

    let downsample3 = ConvBlock::new(128, 256, 3, 2, 1, bias, device);
    let mut res_blocks_3 = Vec::new();
    for _ in 0..8 {
      res_blocks_3.push(ResidualBlock::new(256, device));
    }

    let downsample4 = ConvBlock::new(256, 512, 3, 2, 1, bias, device);
    let mut res_blocks_4 = Vec::new();
    for _ in 0..8 {
      res_blocks_4.push(ResidualBlock::new(512, device));
    }

    let downsample5 = ConvBlock::new(512, 1024, 3, 2, 1, bias, device);
    let mut res_blocks_5 = Vec::new();
    for _ in 0..4 {
      res_blocks_5.push(ResidualBlock::new(1024, device));
    }

    // YOLO Head
    let head_output_channels = self.num_anchors_per_scale * (5 + self.num_classes);

    // Scale 1
    let head_convs_scale1 = vec![
      ConvBlock::new(1024, 512, 1, 1, 0, bias, device),
      ConvBlock::new(512, 1024, 3, 1, 1, bias, device),
      ConvBlock::new(1024, 512, 1, 1, 0, bias, device),
      ConvBlock::new(512, 1024, 3, 1, 1, bias, device),
      ConvBlock::new(1024, 512, 1, 1, 0, bias, device),
      ConvBlock::new(512, 256, 1, 1, 0, bias, device),
    ];

    let detect_layer_scale1 = Conv2dConfig::new([512, head_output_channels], [1, 1])
      .with_bias(bias)
      .init(device);

    // Scale 2
    let head_convs_scale2 = vec![
      ConvBlock::new(512 + 256, 256, 1, 1, 0, bias, device),
      ConvBlock::new(256, 512, 3, 1, 1, bias, device),
      ConvBlock::new(512, 256, 1, 1, 0, bias, device),
      ConvBlock::new(256, 512, 3, 1, 1, bias, device),
      ConvBlock::new(512, 256, 1, 1, 0, bias, device),
      ConvBlock::new(256, 128, 1, 1, 0, bias, device),
    ];

    let detect_layer_scale2 = Conv2dConfig::new([256, head_output_channels], [1, 1])
      .with_bias(bias)
      .init(device);

    // Scale 3
    let head_convs_scale3 = vec![
      ConvBlock::new(256 + 128, 128, 1, 1, 0, bias, device),
      ConvBlock::new(128, 256, 3, 1, 1, bias, device),
      ConvBlock::new(256, 128, 1, 1, 0, bias, device),
      ConvBlock::new(128, 256, 3, 1, 1, bias, device),
      ConvBlock::new(256, 128, 1, 1, 0, bias, device),
    ];

    let detect_layer_scale3 = Conv2dConfig::new([128, head_output_channels], [1, 1])
      .with_bias(bias)
      .init(device);

    YoloV3 {
      conv_initial,
      downsample1,
      res_block_1_1,
      downsample2,
      res_block_2_1,
      res_block_2_2,
      downsample3,
      res_blocks_3,
      downsample4,
      res_blocks_4,
      downsample5,
      res_blocks_5,
      head_convs_scale1,
      detect_layer_scale1,
      head_convs_scale2,
      detect_layer_scale2,
      head_convs_scale3,
      detect_layer_scale3,
    }
  }
}

impl<B: Backend> YoloV3<B> {
  pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    // Backbone
    let x0 = self.conv_initial.forward(input);
    let x1 = self.downsample1.forward(x0);
    let x1 = self.res_block_1_1.forward(x1);

    let x2 = self.downsample2.forward(x1);
    let x2 = self.res_block_2_1.forward(x2);
    let x2 = self.res_block_2_2.forward(x2);

    let mut x3 = self.downsample3.forward(x2);
    for block in &self.res_blocks_3 {
      x3 = block.forward(x3);
    }
    let backbone_out_for_scale3_head = x3.clone();

    let mut x4 = self
      .downsample4
      .forward(backbone_out_for_scale3_head.clone());
    for block in &self.res_blocks_4 {
      x4 = block.forward(x4);
    }
    let backbone_out_for_scale2_head = x4.clone();

    let mut x5 = self
      .downsample5
      .forward(backbone_out_for_scale2_head.clone());
    for block in &self.res_blocks_5 {
      x5 = block.forward(x5);
    }
    let backbone_out_for_scale1_head = x5;

    // YOLO Head
    // Scale 1
    let mut current_features_s1 = backbone_out_for_scale1_head;
    for i in 0..5 {
      current_features_s1 = self.head_convs_scale1[i].forward(current_features_s1);
    }
    let output_scale1 = self
      .detect_layer_scale1
      .forward(current_features_s1.clone());

    let features_s1_processed_for_upsample = self.head_convs_scale1[5].forward(current_features_s1);
    // 简化的上采样实现
    let features_s1_upsampled = features_s1_processed_for_upsample
      .repeat_dim(2, 2)
      .repeat_dim(3, 2);

    // Scale 2
    let concatenated_features_s2 =
      Tensor::cat(vec![features_s1_upsampled, backbone_out_for_scale2_head], 1);

    let mut current_features_s2 = concatenated_features_s2;
    for i in 0..5 {
      current_features_s2 = self.head_convs_scale2[i].forward(current_features_s2);
    }
    let output_scale2 = self
      .detect_layer_scale2
      .forward(current_features_s2.clone());

    let features_s2_processed_for_upsample = self.head_convs_scale2[5].forward(current_features_s2);
    // 简化的上采样实现
    let features_s2_upsampled = features_s2_processed_for_upsample
      .repeat_dim(2, 2)
      .repeat_dim(3, 2);

    // Scale 3
    let concatenated_features_s3 =
      Tensor::cat(vec![features_s2_upsampled, backbone_out_for_scale3_head], 1);

    let mut current_features_s3 = concatenated_features_s3;
    for conv_block in &self.head_convs_scale3 {
      current_features_s3 = conv_block.forward(current_features_s3);
    }
    let output_scale3 = self.detect_layer_scale3.forward(current_features_s3);

    (output_scale1, output_scale2, output_scale3)
  }
}
