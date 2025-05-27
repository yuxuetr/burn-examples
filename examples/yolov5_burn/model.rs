use burn::{
  module::Module,
  nn::{
    BatchNorm, BatchNormConfig, LeakyRelu, LeakyReluConfig, ModuleList, ModuleListState,
    conv::{Conv2d, Conv2dConfig},
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
  },
  tensor::{backend::Backend, Tensor, ops::TensorOps, Device},
};

// ConvBlock Definition (Re-using from YOLOv3, might need adjustments for SiLU activation)
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
  conv: Conv2d<B>,
  bn: BatchNorm<B, 2>,
  // In YOLOv5, SiLU (Swish) is often used. LeakyRelu is a placeholder for now.
  // Consider adding a new Activation module or making it configurable if SiLU is needed.
  activation: LeakyRelu,
}

impl<B: Backend> ConvBlock<B> {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool, // YOLOv5 usually has bias in conv layers
    device: &Device<B>,
  ) -> Self {
    Self {
      conv: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(burn::nn::PaddingConfig2d::Explicit(padding, padding))
        .with_bias(bias)
        .init(device),
      bn: BatchNormConfig::new(out_channels).init(device),
      activation: LeakyReluConfig::new().with_negative_slope(0.1).init(), // Placeholder
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv.forward(input);
    let x = self.bn.forward(x);
    self.activation.forward(x)
  }
}

// Bottleneck for CSP Stages
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
  conv1: ConvBlock<B>,
  conv2: ConvBlock<B>,
  residual: bool,
}

impl<B: Backend> Bottleneck<B> {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    shortcut: bool, // determines if residual connection is used
    expansion: f32, // expansion factor for hidden channels
    device: &Device<B>,
  ) -> Self {
    let hidden_channels = (out_channels as f32 * expansion) as usize;
    Self {
      conv1: ConvBlock::new(in_channels, hidden_channels, 1, 1, 0, true, device),
      conv2: ConvBlock::new(hidden_channels, out_channels, 3, 1, 1, true, device),
      residual: shortcut && in_channels == out_channels,
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv1.forward(input.clone());
    let x = self.conv2.forward(x);
    if self.residual {
      input + x
    } else {
      x
    }
  }
}

// CSPStage (Cross Stage Partial)
#[derive(Module, Debug)]
pub struct CSPStage<B: Backend> {
  conv_downsample: Option<ConvBlock<B>>, // Optional downsampling
  conv_main_path1: ConvBlock<B>,
  conv_main_path2: ConvBlock<B>,
  bottlenecks: ModuleList<Bottleneck<B>>,
  conv_transition: ConvBlock<B>,
}

impl<B: Backend> CSPStage<B> {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    num_bottlenecks: usize,
    downsample: bool,
    expansion: f32, // expansion for bottlenecks
    device: &Device<B>,
  ) -> Self {
    let hidden_channels = (out_channels as f32 / 2.0) as usize; // CSP splits channels

    let conv_downsample = if downsample {
        Some(ConvBlock::new(in_channels, out_channels, 3, 2, 1, true, device))
    } else {
        None
    };
    let current_in_channels = if downsample { out_channels } else { in_channels };

    let conv_main_path1 = ConvBlock::new(current_in_channels, hidden_channels, 1, 1, 0, true, device);
    let conv_main_path2 = ConvBlock::new(current_in_channels, hidden_channels, 1, 1, 0, true, device);
    
    let mut bottlenecks = ModuleList::new();
    for _ in 0..num_bottlenecks {
        bottlenecks.push(Bottleneck::new(hidden_channels, hidden_channels, true, expansion, device));
    }

    let conv_transition = ConvBlock::new(hidden_channels * 2, out_channels, 1, 1, 0, true, device);

    Self {
      conv_downsample,
      conv_main_path1,
      conv_main_path2,
      bottlenecks,
      conv_transition,
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = match &self.conv_downsample {
        Some(conv) => conv.forward(input),
        None => input,
    };

    let x1 = self.conv_main_path1.forward(x.clone());
    let mut x_res = self.conv_main_path2.forward(x);

    for bottleneck in self.bottlenecks.iter() {
        x_res = bottleneck.forward(x_res);
    }
    
    // Concatenate along the channel dimension
    let x_cat = Tensor::cat(vec![x1, x_res], 1);
    self.conv_transition.forward(x_cat)
  }
}


// SPPF (Spatial Pyramid Pooling Fast)
#[derive(Module, Debug)]
pub struct SPPF<B: Backend> {
  conv1: ConvBlock<B>,
  pool1: MaxPool2d,
  pool2: MaxPool2d,
  pool3: MaxPool2d,
  conv2: ConvBlock<B>,
}

impl<B: Backend> SPPF<B> {
  pub fn new(in_channels: usize, out_channels: usize, k: usize, device: &Device<B>) -> Self {
    let hidden_channels = in_channels / 2; // YOLOv5 uses in_channels / 2
    Self {
      conv1: ConvBlock::new(in_channels, hidden_channels, 1, 1, 0, true, device),
      pool1: MaxPool2dConfig::new([k, k]).with_strides([1,1]).with_padding(burn::nn::PaddingConfig2d::Explicit(k/2, k/2)).init(),
      pool2: MaxPool2dConfig::new([k, k]).with_strides([1,1]).with_padding(burn::nn::PaddingConfig2d::Explicit(k/2, k/2)).init(),
      pool3: MaxPool2dConfig::new([k, k]).with_strides([1,1]).with_padding(burn::nn::PaddingConfig2d::Explicit(k/2, k/2)).init(),
      conv2: ConvBlock::new(hidden_channels * 4, out_channels, 1, 1, 0, true, device),
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv1.forward(input);
    let p1 = self.pool1.forward(x.clone());
    let p2 = self.pool2.forward(p1.clone());
    let p3 = self.pool3.forward(p2.clone());
    let concat = Tensor::cat(vec![x, p1, p2, p3], 1);
    self.conv2.forward(concat)
  }
}

// YOLOv5 Backbone (CSPDarknet based)
#[derive(Module, Debug)]
pub struct YoloV5Backbone<B: Backend> {
  stem: ConvBlock<B>, // Initial convolution (e.g., 64 channels, k=6, s=2, p=2)
  stage1: CSPStage<B>, // C3, n=1
  stage2: CSPStage<B>, // C3, n=2, downsample
  stage3: CSPStage<B>, // C3, n=3, downsample
  stage4: CSPStage<B>, // C3, n=1, downsample
  sppf: SPPF<B>,
}

impl<B: Backend> YoloV5Backbone<B> {
    // Parameters for a typical YOLOv5s model
    pub fn new(
        initial_channels: usize, // e.g., 3 for RGB
        base_channels: usize,    // e.g., 64 for YOLOv5s
        depth_multiple: f32,     // scales number of bottlenecks, e.g., 0.33 for YOLOv5s
        channel_multiple: f32,   // scales number of channels, e.g., 0.50 for YOLOv5s
        device: &Device<B>
    ) -> Self {
        let ch = |c: usize| (c as f32 * channel_multiple) as usize;
        let d = |n: usize| (n as f32 * depth_multiple).round() as usize;

        let stem_out_c = ch(base_channels);
        let stem = ConvBlock::new(initial_channels, stem_out_c, 6, 2, 2, true, device); // P1/2

        let s1_in_c = stem_out_c;
        let s1_out_c = ch(base_channels * 2);
        let stage1 = CSPStage::new(s1_in_c, s1_out_c, d(3), true, 0.5, device); // P2/4

        let s2_in_c = s1_out_c;
        let s2_out_c = ch(base_channels * 4);
        let stage2 = CSPStage::new(s2_in_c, s2_out_c, d(6), true, 0.5, device); // P3/8

        let s3_in_c = s2_out_c;
        let s3_out_c = ch(base_channels * 8);
        let stage3 = CSPStage::new(s3_in_c, s3_out_c, d(9), true, 0.5, device); // P4/16

        let s4_in_c = s3_out_c;
        let s4_out_c = ch(base_channels * 16);
        let stage4 = CSPStage::new(s4_in_c, s4_out_c, d(3), true, 0.5, device); // P5/32
        
        let sppf = SPPF::new(s4_out_c, s4_out_c, 5, device);

        Self { stem, stage1, stage2, stage3, stage4, sppf }
    }

    // Returns features from P3, P4, P5 layers
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.stem.forward(input);    // P1
        let x = self.stage1.forward(x);   // P2
        let p3_out = self.stage2.forward(x); // P3/8
        let p4_out = self.stage3.forward(p3_out.clone()); // P4/16
        let p5_out_pre_sppf = self.stage4.forward(p4_out.clone()); // P5/32
        let p5_out = self.sppf.forward(p5_out_pre_sppf);
        (p3_out, p4_out, p5_out)
    }
}


// YOLOv5 Neck (PANet)
#[derive(Module, Debug)]
pub struct YoloV5Neck<B: Backend> {
    // Upsampling path
    upsample_p5_to_p4: burn::nn::conv::ConvTranspose2d<B>, // Upsample
    csp_p4_up: CSPStage<B>, // C3 block for P4 path
    upsample_p4_to_p3: burn::nn::conv::ConvTranspose2d<B>, // Upsample
    csp_p3_up: CSPStage<B>, // C3 block for P3 path

    // Downsampling path
    conv_p3_down: ConvBlock<B>, // Downsample conv for P3 -> P4
    csp_p4_down: CSPStage<B>,   // C3 block for P4 path (bottom-up)
    conv_p4_down: ConvBlock<B>, // Downsample conv for P4 -> P5
    csp_p5_down: CSPStage<B>,   // C3 block for P5 path (bottom-up)
}

impl<B: Backend> YoloV5Neck<B> {
    pub fn new(
        p3_channels: usize, 
        p4_channels: usize, 
        p5_channels: usize,
        depth_multiple: f32, // For CSP stages in neck
        channel_multiple: f32, // For adjusting channels in neck
        device: &Device<B>
    ) -> Self {
        let ch = |c: usize| (c as f32 * channel_multiple) as usize;
        let d = |n: usize| (n as f32 * depth_multiple).round() as usize;

        // Upsampling Path
        // P5 to P4
        let upsample_p5_to_p4 = burn::nn::conv::ConvTranspose2dConfig::new([p5_channels, ch(p4_channels)], [2,2])
            .with_stride([2,2]).with_bias(true).init(device);
        // After upsample, channels become ch(p4_channels). Concatenated with p4_channels from backbone.
        let csp_p4_up = CSPStage::new(ch(p4_channels) + p4_channels, ch(p4_channels), d(3), false, 0.5, device);

        // P4 to P3
        let upsample_p4_to_p3 = burn::nn::conv::ConvTranspose2dConfig::new([ch(p4_channels), ch(p3_channels)], [2,2])
            .with_stride([2,2]).with_bias(true).init(device);
        // After upsample, channels become ch(p3_channels). Concatenated with p3_channels from backbone.
        let csp_p3_up = CSPStage::new(ch(p3_channels) + p3_channels, ch(p3_channels), d(3), false, 0.5, device); // Output N3

        // Downsampling Path
        // P3 to P4
        let conv_p3_down = ConvBlock::new(ch(p3_channels), ch(p3_channels), 3, 2, 1, true, device);
        // Concatenated with output from csp_p4_up (which is ch(p4_channels))
        let csp_p4_down = CSPStage::new(ch(p3_channels) + ch(p4_channels), ch(p4_channels), d(3), false, 0.5, device); // Output N4

        // P4 to P5
        let conv_p4_down = ConvBlock::new(ch(p4_channels), ch(p4_channels), 3, 2, 1, true, device);
        // Concatenated with output from upsample_p5_to_p4 (which is ch(p5_channels), actually p5_channels from backbone after sppf)
        let csp_p5_down = CSPStage::new(ch(p4_channels) + p5_channels, ch(p5_channels), d(3), false, 0.5, device); // Output N5

        Self {
            upsample_p5_to_p4,
            csp_p4_up,
            upsample_p4_to_p3,
            csp_p3_up,
            conv_p3_down,
            csp_p4_down,
            conv_p4_down,
            csp_p5_down,
        }
    }

    // Takes P3, P4, P5 from backbone
    // Returns N3, N4, N5 features for detection heads
    pub fn forward(
        &self,
        p3_in: Tensor<B, 4>, // Backbone P3
        p4_in: Tensor<B, 4>, // Backbone P4
        p5_in: Tensor<B, 4>, // Backbone P5 (after SPPF)
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Top-down path (upsampling)
        let x_p5_up = self.upsample_p5_to_p4.forward(p5_in.clone());
        let x_p4_cat = Tensor::cat(vec![x_p5_up, p4_in.clone()], 1);
        let x_p4_out = self.csp_p4_up.forward(x_p4_cat); // P4 features from neck (for P3 path and P4 detection head)

        let x_p4_up = self.upsample_p4_to_p3.forward(x_p4_out.clone());
        let x_p3_cat = Tensor::cat(vec![x_p4_up, p3_in.clone()], 1);
        let n3_out = self.csp_p3_up.forward(x_p3_cat); // P3 features from neck (N3)

        // Bottom-up path (downsampling)
        let x_p3_down = self.conv_p3_down.forward(n3_out.clone());
        let x_p4_cat_down = Tensor::cat(vec![x_p3_down, x_p4_out.clone()], 1); // x_p4_out is the output of first C3 in neck
        let n4_out = self.csp_p4_down.forward(x_p4_cat_down); // P4 features from neck (N4)

        let x_p4_down = self.conv_p4_down.forward(n4_out.clone());
        let x_p5_cat_down = Tensor::cat(vec![x_p4_down, p5_in.clone()], 1); // p5_in is the output of SPPF from backbone
        let n5_out = self.csp_p5_down.forward(x_p5_cat_down); // P5 features from neck (N5)

        (n3_out, n4_out, n5_out)
    }
}


// Detection Head
#[derive(Module, Debug)]
pub struct DetectionHead<B: Backend> {
  conv: Conv2d<B>, // 1x1 conv to map features to (num_anchors * (5 + num_classes))
}

impl<B: Backend> DetectionHead<B> {
  pub fn new(
    in_channels: usize,
    num_anchors: usize,
    num_classes: usize,
    device: &Device<B>,
  ) -> Self {
    let out_channels = num_anchors * (5 + num_classes); // 5 for (x, y, w, h, conf)
    Self {
      conv: Conv2dConfig::new([in_channels, out_channels], [1, 1]).with_bias(true).init(device),
    }
  }

  pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    self.conv.forward(input)
  }
}

// YOLOv5 Model
#[derive(Module, Debug)]
pub struct YoloV5<B: Backend> {
  backbone: YoloV5Backbone<B>,
  neck: YoloV5Neck<B>,
  detect_head1: DetectionHead<B>, // For P3 features (smallest objects)
  detect_head2: DetectionHead<B>, // For P4 features (medium objects)
  detect_head3: DetectionHead<B>, // For P5 features (largest objects)
  // Anchors and other post-processing attributes might be stored here or passed during forward
}

pub struct YoloV5Config {
  pub num_classes: usize,
  pub num_anchors_per_scale: usize, // typically 3
  // Model variant parameters (e.g., for YOLOv5s, YOLOv5m, etc.)
  pub depth_multiple: f32, // model depth multiple
  pub channel_multiple: f32, // model width multiple
  pub initial_channels: usize, // input channels (e.g. 3 for RGB)
  pub base_backbone_channels: usize, // base channels for backbone stages
}

impl YoloV5Config {
  // Example: YOLOv5s configuration
  pub fn yolov5s(num_classes: usize, num_anchors_per_scale: usize) -> Self {
    Self {
      num_classes,
      num_anchors_per_scale,
      depth_multiple: 0.33,
      channel_multiple: 0.50,
      initial_channels: 3,
      base_backbone_channels: 64,
    }
  }

  pub fn init<B: Backend>(&self, device: &Device<B>) -> YoloV5<B> {
    let ch = |c: usize| (c as f32 * self.channel_multiple) as usize;
    
    let backbone = YoloV5Backbone::new(
        self.initial_channels, 
        self.base_backbone_channels, 
        self.depth_multiple, 
        self.channel_multiple, 
        device
    );

    // Channel sizes from backbone outputs (P3, P4, P5)
    // P3_channels = base_channels * 4 * channel_multiple
    // P4_channels = base_channels * 8 * channel_multiple
    // P5_channels = base_channels * 16 * channel_multiple (after SPPF, still this many channels)
    let p3_c = ch(self.base_backbone_channels * 4);
    let p4_c = ch(self.base_backbone_channels * 8);
    let p5_c = ch(self.base_backbone_channels * 16);

    let neck = YoloV5Neck::new(
        p3_c, 
        p4_c, 
        p5_c, 
        self.depth_multiple, 
        self.channel_multiple, 
        device
    );

    // Detection heads operate on the outputs of the neck (N3, N4, N5)
    // N3_channels = ch(p3_channels) = ch(base_channels * 4)
    // N4_channels = ch(p4_channels) = ch(base_channels * 8)
    // N5_channels = ch(p5_channels) = ch(base_channels * 16)
    let n3_head_in_c = ch(self.base_backbone_channels * 4);
    let n4_head_in_c = ch(self.base_backbone_channels * 8);
    let n5_head_in_c = ch(self.base_backbone_channels * 16);

    let detect_head1 = DetectionHead::new(n3_head_in_c, self.num_anchors_per_scale, self.num_classes, device);
    let detect_head2 = DetectionHead::new(n4_head_in_c, self.num_anchors_per_scale, self.num_classes, device);
    let detect_head3 = DetectionHead::new(n5_head_in_c, self.num_anchors_per_scale, self.num_classes, device);

    YoloV5 {
      backbone,
      neck,
      detect_head1,
      detect_head2,
      detect_head3,
    }
  }
}

impl<B: Backend> YoloV5<B> {
  pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let (p3_features, p4_features, p5_features) = self.backbone.forward(input);
    let (n3_features, n4_features, n5_features) = self.neck.forward(p3_features, p4_features, p5_features);

    let output1 = self.detect_head1.forward(n3_features);
    let output2 = self.detect_head2.forward(n4_features);
    let output3 = self.detect_head3.forward(n5_features);

    (output1, output2, output3) // Corresponds to detections at different scales
  }
}
