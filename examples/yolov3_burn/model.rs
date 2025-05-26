use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        interpolate::{InterpolateNearest, InterpolateNearestConfig}, // Added for upsampling
        BatchNorm, BatchNormConfig, LeakyRelu, LeakyReluConfig,
    },
    tensor::{backend::Backend, Tensor},
};
use burn::config::Config; // Ensure Config is in scope

// ConvBlock Definition (remains unchanged)
#[derive(Module, Debug, Clone, PartialEq)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    activation: LeakyRelu,
}

#[derive(Debug, Clone)]
pub struct ConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
}

impl ConvBlockConfig {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        }
    }
}

impl Config for ConvBlockConfig {
    type Module<B: Backend> = ConvBlock<B>;
    fn init<B: Backend>(&self) -> Self::Module<B> {
        ConvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [self.kernel_size, self.kernel_size])
                .with_stride([self.stride, self.stride])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(self.padding, self.padding))
                .with_bias(self.bias)
                .init(),
            bn: BatchNormConfig::new(self.out_channels).init(),
            activation: LeakyReluConfig::new(0.1).init(),
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        self.activation.forward(x)
    }
}

// ResidualBlock Definition (remains unchanged)
#[derive(Module, Debug, Clone, PartialEq)]
pub struct ResidualBlock<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
}

#[derive(Debug, Clone)]
pub struct ResidualBlockConfig {
    channels: usize,
}

impl ResidualBlockConfig {
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }
}

impl Config for ResidualBlockConfig {
    type Module<B: Backend> = ResidualBlock<B>;
    fn init<B: Backend>(&self) -> Self::Module<B> {
        let half_channels = self.channels / 2; // Darknet typically uses in_channels / 2 for the first conv
        ResidualBlock {
            conv1: ConvBlockConfig::new(self.channels, half_channels, 1, 1, 0, true).init(),
            conv2: ConvBlockConfig::new(half_channels, self.channels, 3, 1, 1, true).init(),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let x = self.conv1.forward(input);
        let x = self.conv2.forward(x);
        x + residual
    }
}


// YoloV3 Model Definition
#[derive(Module, Debug, Clone, PartialEq)]
pub struct YoloV3<B: Backend> {
    // Darknet-53 Backbone
    conv_initial: ConvBlock<B>,      // 3 -> 32
    downsample1: ConvBlock<B>,       // 32 -> 64 (stride 2)
    res_block_1_1: ResidualBlock<B>, // 64 channels

    downsample2: ConvBlock<B>,       // 64 -> 128 (stride 2)
    res_block_2_1: ResidualBlock<B>, // 128 channels
    res_block_2_2: ResidualBlock<B>, // 128 channels

    downsample3: ConvBlock<B>,       // 128 -> 256 (stride 2)
    res_blocks_3: Vec<ResidualBlock<B>>, // 8x ResidualBlocks, 256 channels

    downsample4: ConvBlock<B>,       // 256 -> 512 (stride 2)
    res_blocks_4: Vec<ResidualBlock<B>>, // 8x ResidualBlocks, 512 channels

    downsample5: ConvBlock<B>,       // 512 -> 1024 (stride 2)
    res_blocks_5: Vec<ResidualBlock<B>>, // 4x ResidualBlocks, 1024 channels

    // YOLO Head
    // Scale 1 (large objects, from res_blocks_5 output)
    head_convs_scale1: Vec<ConvBlock<B>>,
    detect_layer_scale1: Conv2d<B>,

    // Scale 2 (medium objects)
    upsample_scale2: InterpolateNearest<B>,
    head_convs_scale2: Vec<ConvBlock<B>>,
    detect_layer_scale2: Conv2d<B>,

    // Scale 3 (small objects)
    upsample_scale3: InterpolateNearest<B>,
    head_convs_scale3: Vec<ConvBlock<B>>,
    detect_layer_scale3: Conv2d<B>,
}

#[derive(Debug, Clone)]
pub struct YoloV3Config {
    num_classes: usize,
    num_anchors_per_scale: usize, // Typically 3 for YOLOv3
}

impl YoloV3Config {
    pub fn new(num_classes: usize, num_anchors_per_scale: usize) -> Self {
        Self { num_classes, num_anchors_per_scale }
    }
}

impl Config for YoloV3Config {
    type Module<B: Backend> = YoloV3<B>;

    fn init<B: Backend>(&self) -> YoloV3<B> {
        let bias = true; // Common setting for YOLO conv layers

        // Darknet-53 Backbone
        let conv_initial = ConvBlockConfig::new(3, 32, 3, 1, 1, bias).init();
        let downsample1 = ConvBlockConfig::new(32, 64, 3, 2, 1, bias).init();
        let res_block_1_1 = ResidualBlockConfig::new(64).init();

        let downsample2 = ConvBlockConfig::new(64, 128, 3, 2, 1, bias).init();
        let res_block_2_1 = ResidualBlockConfig::new(128).init();
        let res_block_2_2 = ResidualBlockConfig::new(128).init();

        let downsample3 = ConvBlockConfig::new(128, 256, 3, 2, 1, bias).init();
        let mut res_blocks_3 = Vec::new();
        for _ in 0..8 {
            res_blocks_3.push(ResidualBlockConfig::new(256).init());
        }

        let downsample4 = ConvBlockConfig::new(256, 512, 3, 2, 1, bias).init();
        let mut res_blocks_4 = Vec::new();
        for _ in 0..8 {
            res_blocks_4.push(ResidualBlockConfig::new(512).init());
        }

        let downsample5 = ConvBlockConfig::new(512, 1024, 3, 2, 1, bias).init();
        let mut res_blocks_5 = Vec::new();
        for _ in 0..4 {
            res_blocks_5.push(ResidualBlockConfig::new(1024).init());
        }

        // YOLO Head
        let head_output_channels = self.num_anchors_per_scale * (5 + self.num_classes);

        // Scale 1 (from 1024 channels)
        let mut head_convs_scale1 = Vec::new();
        head_convs_scale1.push(ConvBlockConfig::new(1024, 512, 1, 1, 0, bias).init());
        head_convs_scale1.push(ConvBlockConfig::new(512, 1024, 3, 1, 1, bias).init());
        head_convs_scale1.push(ConvBlockConfig::new(1024, 512, 1, 1, 0, bias).init());
        head_convs_scale1.push(ConvBlockConfig::new(512, 1024, 3, 1, 1, bias).init());
        head_convs_scale1.push(ConvBlockConfig::new(1024, 512, 1, 1, 0, bias).init());
        let detect_layer_scale1_input_channels = 512;
        let detect_layer_scale1 = Conv2dConfig::new(
            [detect_layer_scale1_input_channels, head_output_channels],
            [1, 1], // 1x1 convolution for detection
        ).with_bias(bias).init();


        // Scale 2 (from 512 backbone + 256 upsampled from scale 1's 512 channels)
        // Input to head_convs_scale2 will be 512 (backbone) + 256 (upsampled from 512 after conv) = 768
        let upsample_scale2 = InterpolateNearestConfig::new(2).init(); // Scale factor 2
        let conv_before_upsample_scale2 = ConvBlockConfig::new(512, 256, 1, 1, 0, bias).init(); // Added to match common YOLO patterns
        head_convs_scale1.push(conv_before_upsample_scale2); // Technically part of scale 1 processing before upsampling for scale 2

        let mut head_convs_scale2 = Vec::new();
        head_convs_scale2.push(ConvBlockConfig::new(512 + 256, 256, 1, 1, 0, bias).init());
        head_convs_scale2.push(ConvBlockConfig::new(256, 512, 3, 1, 1, bias).init());
        head_convs_scale2.push(ConvBlockConfig::new(512, 256, 1, 1, 0, bias).init());
        head_convs_scale2.push(ConvBlockConfig::new(256, 512, 3, 1, 1, bias).init());
        head_convs_scale2.push(ConvBlockConfig::new(512, 256, 1, 1, 0, bias).init());
        let detect_layer_scale2_input_channels = 256;
        let detect_layer_scale2 = Conv2dConfig::new(
            [detect_layer_scale2_input_channels, head_output_channels],
            [1, 1],
        ).with_bias(bias).init();

        // Scale 3 (from 256 backbone + 128 upsampled from scale 2's 256 channels)
        // Input to head_convs_scale3 will be 256 (backbone) + 128 (upsampled from 256 after conv) = 384
        let upsample_scale3 = InterpolateNearestConfig::new(2).init(); // Scale factor 2
        let conv_before_upsample_scale3 = ConvBlockConfig::new(256, 128, 1, 1, 0, bias).init(); // Added
        head_convs_scale2.push(conv_before_upsample_scale3); // Part of scale 2 processing for scale 3

        let mut head_convs_scale3 = Vec::new();
        head_convs_scale3.push(ConvBlockConfig::new(256 + 128, 128, 1, 1, 0, bias).init());
        head_convs_scale3.push(ConvBlockConfig::new(128, 256, 3, 1, 1, bias).init());
        head_convs_scale3.push(ConvBlockConfig::new(256, 128, 1, 1, 0, bias).init());
        head_convs_scale3.push(ConvBlockConfig::new(128, 256, 3, 1, 1, bias).init());
        head_convs_scale3.push(ConvBlockConfig::new(256, 128, 1, 1, 0, bias).init());
        let detect_layer_scale3_input_channels = 128;
        let detect_layer_scale3 = Conv2dConfig::new(
            [detect_layer_scale3_input_channels, head_output_channels],
            [1, 1],
        ).with_bias(bias).init();


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
            upsample_scale2,
            head_convs_scale2,
            detect_layer_scale2,
            upsample_scale3,
            head_convs_scale3,
            detect_layer_scale3,
        }
    }
}

impl<B: Backend> YoloV3<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Backbone
        // The backbone structure is:
        // conv_initial (3->32)
        // downsample1 (32->64), res_block_1_1 (1x res on 64)
        // downsample2 (64->128), res_block_2_1, res_block_2_2 (2x res on 128)
        // downsample3 (128->256), res_blocks_3 (8x res on 256) <- output for scale3 head (smallest objects / largest feature map)
        // downsample4 (256->512), res_blocks_4 (8x res on 512) <- output for scale2 head (medium objects / medium feature map)
        // downsample5 (512->1024), res_blocks_5 (4x res on 1024) <- output for scale1 head (largest objects / smallest feature map)

        let x0 = self.conv_initial.forward(input);
        let x1 = self.downsample1.forward(x0);
        let x1 = self.res_block_1_1.forward(x1);

        let x2 = self.downsample2.forward(x1);
        let x2 = self.res_block_2_1.forward(x2);
        let x2 = self.res_block_2_2.forward(x2); // Output from here is 128 channels. This is `scale1_features_raw` (256 channels in subtask desc, but YOLO connects earlier stages for larger feature maps)
                                                 // For YOLOv3, the connections are typically:
                                                 // - Smallest objects (largest feature map): from after 8th residual block group (256 ch)
                                                 // - Medium objects: from after 16th residual block group (512 ch)
                                                 // - Large objects: from final backbone layer (1024 ch)

        let mut x3 = self.downsample3.forward(x2); // 128 -> 256
        for block in &self.res_blocks_3 {
            x3 = block.forward(x3);
        }
        let backbone_out_for_scale3_head = x3; // 256 channels, for smallest objects detection (largest feature map)

        let mut x4 = self.downsample4.forward(backbone_out_for_scale3_head); // 256 -> 512
        for block in &self.res_blocks_4 {
            x4 = block.forward(x4);
        }
        let backbone_out_for_scale2_head = x4; // 512 channels, for medium objects detection

        let mut x5 = self.downsample5.forward(backbone_out_for_scale2_head); // 512 -> 1024
        for block in &self.res_blocks_5 {
            x5 = block.forward(x5);
        }
        let backbone_out_for_scale1_head = x5; // 1024 channels, for largest objects detection (smallest feature map)


        // YOLO Head
        // Scale 1 (Large objects, processes backbone_out_for_scale1_head which is 1024 channels)
        let mut current_features_s1 = backbone_out_for_scale1_head;
        // self.head_convs_scale1 has 6 blocks. The first 5 are for this scale's detection path.
        // The 6th block (index 5) is ConvBlockConfig::new(512, 256, 1, 1, 0, bias) and is for preparing features for upsampling to scale 2.
        for i in 0..5 { // Process through the first 5 ConvBlocks for scale 1 detection
            current_features_s1 = self.head_convs_scale1[i].forward(current_features_s1);
        }
        let output_scale1 = self.detect_layer_scale1.forward(current_features_s1.clone()); // current_features_s1 is 512 channels here

        // Prepare for Scale 2: take the 512ch output from ^, pass through the 6th conv block in head_convs_scale1
        let features_s1_processed_for_upsample = self.head_convs_scale1[5].forward(current_features_s1); // 512 -> 256 channels
        let features_s1_upsampled = self.upsample_scale2.forward(features_s1_processed_for_upsample); // Upsample 256ch features


        // Scale 2 (Medium objects)
        // Concatenate upsampled features (256 ch) with backbone output (512 ch from backbone_out_for_scale2_head)
        // Resulting tensor should have 256 + 512 = 768 channels.
        let concatenated_features_s2 = Tensor::cat(
            &[features_s1_upsampled, backbone_out_for_scale2_head],
            1, // Concatenate along the channel dimension
        );

        let mut current_features_s2 = concatenated_features_s2;
        // self.head_convs_scale2 has 6 blocks. The first 5 are for this scale's detection path (input 768 ch).
        // The 6th block (index 5) is ConvBlockConfig::new(256, 128, 1, 1, 0, bias) and is for preparing features for upsampling to scale 3.
        for i in 0..5 { // Process through the first 5 ConvBlocks for scale 2 detection
            current_features_s2 = self.head_convs_scale2[i].forward(current_features_s2);
        }
        let output_scale2 = self.detect_layer_scale2.forward(current_features_s2.clone()); // current_features_s2 is 256 channels here

        // Prepare for Scale 3: take the 256ch output from ^, pass through the 6th conv block in head_convs_scale2
        let features_s2_processed_for_upsample = self.head_convs_scale2[5].forward(current_features_s2); // 256 -> 128 channels
        let features_s2_upsampled = self.upsample_scale3.forward(features_s2_processed_for_upsample); // Upsample 128ch features


        // Scale 3 (Smallest objects)
        // Concatenate upsampled features (128 ch) with backbone output (256 ch from backbone_out_for_scale3_head)
        // Resulting tensor should have 128 + 256 = 384 channels.
        let concatenated_features_s3 = Tensor::cat(
            &[features_s2_upsampled, backbone_out_for_scale3_head],
            1, // Concatenate along the channel dimension
        );

        let mut current_features_s3 = concatenated_features_s3;
        // self.head_convs_scale3 has 5 blocks. All are for this scale's detection path.
        for conv_block in &self.head_convs_scale3 {
            current_features_s3 = conv_block.forward(current_features_s3);
        }
        let output_scale3 = self.detect_layer_scale3.forward(current_features_s3); // current_features_s3 is 128 channels here

        (output_scale1, output_scale2, output_scale3)
    }
}
