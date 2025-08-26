import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_1_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_1_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_3_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_3_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_3_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_3_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_4_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_4_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_5_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_5_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_6_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_6_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_7_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_7_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_8_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_8_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_9_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_9_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_10_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_10_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_11_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_11_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_12_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_12_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_13_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_13_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_14_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_14_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_15_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_15_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_16_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_16_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_17_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_17_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_18_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_18_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_19_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_19_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_20_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_20_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_21_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_21_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_22_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_22_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_23_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_23_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_24_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_24_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_25_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_25_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_26_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_26_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_7_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_7_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_classifier_modules_0_parameters_weight_ = (
            L_self_modules_classifier_modules_0_parameters_weight_
        )
        l_self_modules_classifier_modules_0_parameters_bias_ = (
            L_self_modules_classifier_modules_0_parameters_bias_
        )
        l_self_modules_classifier_modules_2_parameters_weight_ = (
            L_self_modules_classifier_modules_2_parameters_weight_
        )
        l_self_modules_classifier_modules_2_parameters_bias_ = (
            L_self_modules_classifier_modules_2_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_modules_0_parameters_weight_,
            l_self_modules_features_modules_0_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_0_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_0_parameters_bias_ = None
        x = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_1 = torch.nn.functional.layer_norm(
            x,
            (128,),
            l_self_modules_features_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_1_parameters_bias_,
            1e-06,
        )
        x = (
            l_self_modules_features_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_1_parameters_bias_ = None
        x_2 = x_1.permute(0, 3, 1, 2)
        x_1 = None
        input_2 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_3 = torch.permute(input_2, [0, 2, 3, 1])
        input_2 = None
        input_4 = torch.nn.functional.layer_norm(
            input_3,
            (128,),
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_3 = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_5 = torch._C._nn.linear(
            input_4,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_4 = l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_6 = torch._C._nn.gelu(input_5, approximate="none")
        input_5 = None
        input_7 = torch._C._nn.linear(
            input_6,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_6 = l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_8 = torch.permute(input_7, [0, 3, 1, 2])
        input_7 = None
        result = (
            l_self_modules_features_modules_1_modules_0_parameters_layer_scale_
            * input_8
        )
        l_self_modules_features_modules_1_modules_0_parameters_layer_scale_ = (
            input_8
        ) = None
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        result += x_2
        result_1 = result
        result = x_2 = None
        input_9 = torch.conv2d(
            result_1,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_10 = torch.permute(input_9, [0, 2, 3, 1])
        input_9 = None
        input_11 = torch.nn.functional.layer_norm(
            input_10,
            (128,),
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_10 = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_12 = torch._C._nn.linear(
            input_11,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_11 = l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_13 = torch._C._nn.gelu(input_12, approximate="none")
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_13 = l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_15 = torch.permute(input_14, [0, 3, 1, 2])
        input_14 = None
        result_2 = (
            l_self_modules_features_modules_1_modules_1_parameters_layer_scale_
            * input_15
        )
        l_self_modules_features_modules_1_modules_1_parameters_layer_scale_ = (
            input_15
        ) = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        result_2 += result_1
        result_3 = result_2
        result_2 = result_1 = None
        input_16 = torch.conv2d(
            result_3,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_17 = torch.permute(input_16, [0, 2, 3, 1])
        input_16 = None
        input_18 = torch.nn.functional.layer_norm(
            input_17,
            (128,),
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_17 = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_18 = l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        input_21 = torch._C._nn.linear(
            input_20,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_20 = l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_22 = torch.permute(input_21, [0, 3, 1, 2])
        input_21 = None
        result_4 = (
            l_self_modules_features_modules_1_modules_2_parameters_layer_scale_
            * input_22
        )
        l_self_modules_features_modules_1_modules_2_parameters_layer_scale_ = (
            input_22
        ) = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        result_4 += result_3
        result_5 = result_4
        result_4 = result_3 = None
        x_3 = result_5.permute(0, 2, 3, 1)
        result_5 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (128,),
            l_self_modules_features_modules_2_modules_0_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_parameters_bias_,
            1e-06,
        )
        x_3 = (
            l_self_modules_features_modules_2_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_0_parameters_bias_ = None
        x_5 = x_4.permute(0, 3, 1, 2)
        x_4 = None
        input_23 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_features_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_1_parameters_bias_ = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_25 = torch.permute(input_24, [0, 2, 3, 1])
        input_24 = None
        input_26 = torch.nn.functional.layer_norm(
            input_25,
            (256,),
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_25 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_27 = torch._C._nn.linear(
            input_26,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_26 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_28 = torch._C._nn.gelu(input_27, approximate="none")
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_28 = l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_30 = torch.permute(input_29, [0, 3, 1, 2])
        input_29 = None
        result_6 = (
            l_self_modules_features_modules_3_modules_0_parameters_layer_scale_
            * input_30
        )
        l_self_modules_features_modules_3_modules_0_parameters_layer_scale_ = (
            input_30
        ) = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        result_6 += input_23
        result_7 = result_6
        result_6 = input_23 = None
        input_31 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_32 = torch.permute(input_31, [0, 2, 3, 1])
        input_31 = None
        input_33 = torch.nn.functional.layer_norm(
            input_32,
            (256,),
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_32 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_33 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_35 = torch._C._nn.gelu(input_34, approximate="none")
        input_34 = None
        input_36 = torch._C._nn.linear(
            input_35,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_35 = l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_37 = torch.permute(input_36, [0, 3, 1, 2])
        input_36 = None
        result_8 = (
            l_self_modules_features_modules_3_modules_1_parameters_layer_scale_
            * input_37
        )
        l_self_modules_features_modules_3_modules_1_parameters_layer_scale_ = (
            input_37
        ) = None
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        result_8 += result_7
        result_9 = result_8
        result_8 = result_7 = None
        input_38 = torch.conv2d(
            result_9,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_39 = torch.permute(input_38, [0, 2, 3, 1])
        input_38 = None
        input_40 = torch.nn.functional.layer_norm(
            input_39,
            (256,),
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_39 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            input_40,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_40 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch._C._nn.linear(
            input_42,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_42 = l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_44 = torch.permute(input_43, [0, 3, 1, 2])
        input_43 = None
        result_10 = (
            l_self_modules_features_modules_3_modules_2_parameters_layer_scale_
            * input_44
        )
        l_self_modules_features_modules_3_modules_2_parameters_layer_scale_ = (
            input_44
        ) = None
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        result_10 += result_9
        result_11 = result_10
        result_10 = result_9 = None
        x_6 = result_11.permute(0, 2, 3, 1)
        result_11 = None
        x_7 = torch.nn.functional.layer_norm(
            x_6,
            (256,),
            l_self_modules_features_modules_4_modules_0_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_parameters_bias_,
            1e-06,
        )
        x_6 = (
            l_self_modules_features_modules_4_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_0_parameters_bias_ = None
        x_8 = x_7.permute(0, 3, 1, 2)
        x_7 = None
        input_45 = torch.conv2d(
            x_8,
            l_self_modules_features_modules_4_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_features_modules_4_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_1_parameters_bias_ = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_47 = torch.permute(input_46, [0, 2, 3, 1])
        input_46 = None
        input_48 = torch.nn.functional.layer_norm(
            input_47,
            (512,),
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_47 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_48 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        input_51 = torch._C._nn.linear(
            input_50,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_50 = l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_52 = torch.permute(input_51, [0, 3, 1, 2])
        input_51 = None
        result_12 = (
            l_self_modules_features_modules_5_modules_0_parameters_layer_scale_
            * input_52
        )
        l_self_modules_features_modules_5_modules_0_parameters_layer_scale_ = (
            input_52
        ) = None
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        result_12 += input_45
        result_13 = result_12
        result_12 = input_45 = None
        input_53 = torch.conv2d(
            result_13,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_54 = torch.permute(input_53, [0, 2, 3, 1])
        input_53 = None
        input_55 = torch.nn.functional.layer_norm(
            input_54,
            (512,),
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_54 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            input_55,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_55 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch._C._nn.linear(
            input_57,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_57 = l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_59 = torch.permute(input_58, [0, 3, 1, 2])
        input_58 = None
        result_14 = (
            l_self_modules_features_modules_5_modules_1_parameters_layer_scale_
            * input_59
        )
        l_self_modules_features_modules_5_modules_1_parameters_layer_scale_ = (
            input_59
        ) = None
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        result_14 += result_13
        result_15 = result_14
        result_14 = result_13 = None
        input_60 = torch.conv2d(
            result_15,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_61 = torch.permute(input_60, [0, 2, 3, 1])
        input_60 = None
        input_62 = torch.nn.functional.layer_norm(
            input_61,
            (512,),
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_61 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_63 = torch._C._nn.linear(
            input_62,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_62 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        input_65 = torch._C._nn.linear(
            input_64,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_64 = l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_66 = torch.permute(input_65, [0, 3, 1, 2])
        input_65 = None
        result_16 = (
            l_self_modules_features_modules_5_modules_2_parameters_layer_scale_
            * input_66
        )
        l_self_modules_features_modules_5_modules_2_parameters_layer_scale_ = (
            input_66
        ) = None
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        result_16 += result_15
        result_17 = result_16
        result_16 = result_15 = None
        input_67 = torch.conv2d(
            result_17,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_ = (None)
        input_68 = torch.permute(input_67, [0, 2, 3, 1])
        input_67 = None
        input_69 = torch.nn.functional.layer_norm(
            input_68,
            (512,),
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_68 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_ = (None)
        input_70 = torch._C._nn.linear(
            input_69,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_,
        )
        input_69 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_ = (None)
        input_71 = torch._C._nn.gelu(input_70, approximate="none")
        input_70 = None
        input_72 = torch._C._nn.linear(
            input_71,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_,
        )
        input_71 = l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_ = (None)
        input_73 = torch.permute(input_72, [0, 3, 1, 2])
        input_72 = None
        result_18 = (
            l_self_modules_features_modules_5_modules_3_parameters_layer_scale_
            * input_73
        )
        l_self_modules_features_modules_5_modules_3_parameters_layer_scale_ = (
            input_73
        ) = None
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        result_18 += result_17
        result_19 = result_18
        result_18 = result_17 = None
        input_74 = torch.conv2d(
            result_19,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_ = (None)
        input_75 = torch.permute(input_74, [0, 2, 3, 1])
        input_74 = None
        input_76 = torch.nn.functional.layer_norm(
            input_75,
            (512,),
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_75 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_ = (None)
        input_77 = torch._C._nn.linear(
            input_76,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_,
        )
        input_76 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_ = (None)
        input_78 = torch._C._nn.gelu(input_77, approximate="none")
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_,
        )
        input_78 = l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_ = (None)
        input_80 = torch.permute(input_79, [0, 3, 1, 2])
        input_79 = None
        result_20 = (
            l_self_modules_features_modules_5_modules_4_parameters_layer_scale_
            * input_80
        )
        l_self_modules_features_modules_5_modules_4_parameters_layer_scale_ = (
            input_80
        ) = None
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        result_20 += result_19
        result_21 = result_20
        result_20 = result_19 = None
        input_81 = torch.conv2d(
            result_21,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_ = (None)
        input_82 = torch.permute(input_81, [0, 2, 3, 1])
        input_81 = None
        input_83 = torch.nn.functional.layer_norm(
            input_82,
            (512,),
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_82 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_ = (None)
        input_84 = torch._C._nn.linear(
            input_83,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_,
        )
        input_83 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_ = (None)
        input_85 = torch._C._nn.gelu(input_84, approximate="none")
        input_84 = None
        input_86 = torch._C._nn.linear(
            input_85,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_,
        )
        input_85 = l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_ = (None)
        input_87 = torch.permute(input_86, [0, 3, 1, 2])
        input_86 = None
        result_22 = (
            l_self_modules_features_modules_5_modules_5_parameters_layer_scale_
            * input_87
        )
        l_self_modules_features_modules_5_modules_5_parameters_layer_scale_ = (
            input_87
        ) = None
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        result_22 += result_21
        result_23 = result_22
        result_22 = result_21 = None
        input_88 = torch.conv2d(
            result_23,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_ = (None)
        input_89 = torch.permute(input_88, [0, 2, 3, 1])
        input_88 = None
        input_90 = torch.nn.functional.layer_norm(
            input_89,
            (512,),
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_89 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_ = (None)
        input_91 = torch._C._nn.linear(
            input_90,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_,
        )
        input_90 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        input_93 = torch._C._nn.linear(
            input_92,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_,
        )
        input_92 = l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_ = (None)
        input_94 = torch.permute(input_93, [0, 3, 1, 2])
        input_93 = None
        result_24 = (
            l_self_modules_features_modules_5_modules_6_parameters_layer_scale_
            * input_94
        )
        l_self_modules_features_modules_5_modules_6_parameters_layer_scale_ = (
            input_94
        ) = None
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        result_24 += result_23
        result_25 = result_24
        result_24 = result_23 = None
        input_95 = torch.conv2d(
            result_25,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_ = (None)
        input_96 = torch.permute(input_95, [0, 2, 3, 1])
        input_95 = None
        input_97 = torch.nn.functional.layer_norm(
            input_96,
            (512,),
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_96 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_ = (None)
        input_98 = torch._C._nn.linear(
            input_97,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_,
        )
        input_97 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_ = (None)
        input_99 = torch._C._nn.gelu(input_98, approximate="none")
        input_98 = None
        input_100 = torch._C._nn.linear(
            input_99,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_,
        )
        input_99 = l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_ = (None)
        input_101 = torch.permute(input_100, [0, 3, 1, 2])
        input_100 = None
        result_26 = (
            l_self_modules_features_modules_5_modules_7_parameters_layer_scale_
            * input_101
        )
        l_self_modules_features_modules_5_modules_7_parameters_layer_scale_ = (
            input_101
        ) = None
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        result_26 += result_25
        result_27 = result_26
        result_26 = result_25 = None
        input_102 = torch.conv2d(
            result_27,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_ = (None)
        input_103 = torch.permute(input_102, [0, 2, 3, 1])
        input_102 = None
        input_104 = torch.nn.functional.layer_norm(
            input_103,
            (512,),
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_103 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_ = (None)
        input_105 = torch._C._nn.linear(
            input_104,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_,
        )
        input_104 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_ = (None)
        input_106 = torch._C._nn.gelu(input_105, approximate="none")
        input_105 = None
        input_107 = torch._C._nn.linear(
            input_106,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_,
        )
        input_106 = l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_ = (None)
        input_108 = torch.permute(input_107, [0, 3, 1, 2])
        input_107 = None
        result_28 = (
            l_self_modules_features_modules_5_modules_8_parameters_layer_scale_
            * input_108
        )
        l_self_modules_features_modules_5_modules_8_parameters_layer_scale_ = (
            input_108
        ) = None
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        result_28 += result_27
        result_29 = result_28
        result_28 = result_27 = None
        input_109 = torch.conv2d(
            result_29,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_ = (None)
        input_110 = torch.permute(input_109, [0, 2, 3, 1])
        input_109 = None
        input_111 = torch.nn.functional.layer_norm(
            input_110,
            (512,),
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_110 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_ = (None)
        input_112 = torch._C._nn.linear(
            input_111,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_,
        )
        input_111 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_ = (None)
        input_113 = torch._C._nn.gelu(input_112, approximate="none")
        input_112 = None
        input_114 = torch._C._nn.linear(
            input_113,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_,
        )
        input_113 = l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_ = (None)
        input_115 = torch.permute(input_114, [0, 3, 1, 2])
        input_114 = None
        result_30 = (
            l_self_modules_features_modules_5_modules_9_parameters_layer_scale_
            * input_115
        )
        l_self_modules_features_modules_5_modules_9_parameters_layer_scale_ = (
            input_115
        ) = None
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        result_30 += result_29
        result_31 = result_30
        result_30 = result_29 = None
        input_116 = torch.conv2d(
            result_31,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_ = (None)
        input_117 = torch.permute(input_116, [0, 2, 3, 1])
        input_116 = None
        input_118 = torch.nn.functional.layer_norm(
            input_117,
            (512,),
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_117 = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_ = (None)
        input_119 = torch._C._nn.linear(
            input_118,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_,
        )
        input_118 = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_ = (None)
        input_120 = torch._C._nn.gelu(input_119, approximate="none")
        input_119 = None
        input_121 = torch._C._nn.linear(
            input_120,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_,
        )
        input_120 = l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_ = (None)
        input_122 = torch.permute(input_121, [0, 3, 1, 2])
        input_121 = None
        result_32 = (
            l_self_modules_features_modules_5_modules_10_parameters_layer_scale_
            * input_122
        )
        l_self_modules_features_modules_5_modules_10_parameters_layer_scale_ = (
            input_122
        ) = None
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        result_32 += result_31
        result_33 = result_32
        result_32 = result_31 = None
        input_123 = torch.conv2d(
            result_33,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_ = (None)
        input_124 = torch.permute(input_123, [0, 2, 3, 1])
        input_123 = None
        input_125 = torch.nn.functional.layer_norm(
            input_124,
            (512,),
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_124 = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_ = (None)
        input_126 = torch._C._nn.linear(
            input_125,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_,
        )
        input_125 = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_ = (None)
        input_127 = torch._C._nn.gelu(input_126, approximate="none")
        input_126 = None
        input_128 = torch._C._nn.linear(
            input_127,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_,
        )
        input_127 = l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_ = (None)
        input_129 = torch.permute(input_128, [0, 3, 1, 2])
        input_128 = None
        result_34 = (
            l_self_modules_features_modules_5_modules_11_parameters_layer_scale_
            * input_129
        )
        l_self_modules_features_modules_5_modules_11_parameters_layer_scale_ = (
            input_129
        ) = None
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        result_34 += result_33
        result_35 = result_34
        result_34 = result_33 = None
        input_130 = torch.conv2d(
            result_35,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_ = (None)
        input_131 = torch.permute(input_130, [0, 2, 3, 1])
        input_130 = None
        input_132 = torch.nn.functional.layer_norm(
            input_131,
            (512,),
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_131 = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_ = (None)
        input_133 = torch._C._nn.linear(
            input_132,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_,
        )
        input_132 = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_ = (None)
        input_134 = torch._C._nn.gelu(input_133, approximate="none")
        input_133 = None
        input_135 = torch._C._nn.linear(
            input_134,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_,
        )
        input_134 = l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_ = (None)
        input_136 = torch.permute(input_135, [0, 3, 1, 2])
        input_135 = None
        result_36 = (
            l_self_modules_features_modules_5_modules_12_parameters_layer_scale_
            * input_136
        )
        l_self_modules_features_modules_5_modules_12_parameters_layer_scale_ = (
            input_136
        ) = None
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        result_36 += result_35
        result_37 = result_36
        result_36 = result_35 = None
        input_137 = torch.conv2d(
            result_37,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_ = (None)
        input_138 = torch.permute(input_137, [0, 2, 3, 1])
        input_137 = None
        input_139 = torch.nn.functional.layer_norm(
            input_138,
            (512,),
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_138 = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_ = (None)
        input_140 = torch._C._nn.linear(
            input_139,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_,
        )
        input_139 = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_ = (None)
        input_141 = torch._C._nn.gelu(input_140, approximate="none")
        input_140 = None
        input_142 = torch._C._nn.linear(
            input_141,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_,
        )
        input_141 = l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_ = (None)
        input_143 = torch.permute(input_142, [0, 3, 1, 2])
        input_142 = None
        result_38 = (
            l_self_modules_features_modules_5_modules_13_parameters_layer_scale_
            * input_143
        )
        l_self_modules_features_modules_5_modules_13_parameters_layer_scale_ = (
            input_143
        ) = None
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        result_38 += result_37
        result_39 = result_38
        result_38 = result_37 = None
        input_144 = torch.conv2d(
            result_39,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_ = (None)
        input_145 = torch.permute(input_144, [0, 2, 3, 1])
        input_144 = None
        input_146 = torch.nn.functional.layer_norm(
            input_145,
            (512,),
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_145 = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_ = (None)
        input_147 = torch._C._nn.linear(
            input_146,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_,
        )
        input_146 = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_ = (None)
        input_148 = torch._C._nn.gelu(input_147, approximate="none")
        input_147 = None
        input_149 = torch._C._nn.linear(
            input_148,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_,
        )
        input_148 = l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_ = (None)
        input_150 = torch.permute(input_149, [0, 3, 1, 2])
        input_149 = None
        result_40 = (
            l_self_modules_features_modules_5_modules_14_parameters_layer_scale_
            * input_150
        )
        l_self_modules_features_modules_5_modules_14_parameters_layer_scale_ = (
            input_150
        ) = None
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        result_40 += result_39
        result_41 = result_40
        result_40 = result_39 = None
        input_151 = torch.conv2d(
            result_41,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_ = (None)
        input_152 = torch.permute(input_151, [0, 2, 3, 1])
        input_151 = None
        input_153 = torch.nn.functional.layer_norm(
            input_152,
            (512,),
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_152 = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_ = (None)
        input_154 = torch._C._nn.linear(
            input_153,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_,
        )
        input_153 = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_ = (None)
        input_155 = torch._C._nn.gelu(input_154, approximate="none")
        input_154 = None
        input_156 = torch._C._nn.linear(
            input_155,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_,
        )
        input_155 = l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_ = (None)
        input_157 = torch.permute(input_156, [0, 3, 1, 2])
        input_156 = None
        result_42 = (
            l_self_modules_features_modules_5_modules_15_parameters_layer_scale_
            * input_157
        )
        l_self_modules_features_modules_5_modules_15_parameters_layer_scale_ = (
            input_157
        ) = None
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        result_42 += result_41
        result_43 = result_42
        result_42 = result_41 = None
        input_158 = torch.conv2d(
            result_43,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_ = (None)
        input_159 = torch.permute(input_158, [0, 2, 3, 1])
        input_158 = None
        input_160 = torch.nn.functional.layer_norm(
            input_159,
            (512,),
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_159 = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_ = (None)
        input_161 = torch._C._nn.linear(
            input_160,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_,
        )
        input_160 = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_ = (None)
        input_162 = torch._C._nn.gelu(input_161, approximate="none")
        input_161 = None
        input_163 = torch._C._nn.linear(
            input_162,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_,
        )
        input_162 = l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_ = (None)
        input_164 = torch.permute(input_163, [0, 3, 1, 2])
        input_163 = None
        result_44 = (
            l_self_modules_features_modules_5_modules_16_parameters_layer_scale_
            * input_164
        )
        l_self_modules_features_modules_5_modules_16_parameters_layer_scale_ = (
            input_164
        ) = None
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        result_44 += result_43
        result_45 = result_44
        result_44 = result_43 = None
        input_165 = torch.conv2d(
            result_45,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_ = (None)
        input_166 = torch.permute(input_165, [0, 2, 3, 1])
        input_165 = None
        input_167 = torch.nn.functional.layer_norm(
            input_166,
            (512,),
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_166 = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_ = (None)
        input_168 = torch._C._nn.linear(
            input_167,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_,
        )
        input_167 = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_ = (None)
        input_169 = torch._C._nn.gelu(input_168, approximate="none")
        input_168 = None
        input_170 = torch._C._nn.linear(
            input_169,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_,
        )
        input_169 = l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_ = (None)
        input_171 = torch.permute(input_170, [0, 3, 1, 2])
        input_170 = None
        result_46 = (
            l_self_modules_features_modules_5_modules_17_parameters_layer_scale_
            * input_171
        )
        l_self_modules_features_modules_5_modules_17_parameters_layer_scale_ = (
            input_171
        ) = None
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        result_46 += result_45
        result_47 = result_46
        result_46 = result_45 = None
        input_172 = torch.conv2d(
            result_47,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_ = (None)
        input_173 = torch.permute(input_172, [0, 2, 3, 1])
        input_172 = None
        input_174 = torch.nn.functional.layer_norm(
            input_173,
            (512,),
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_173 = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_ = (None)
        input_175 = torch._C._nn.linear(
            input_174,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_,
        )
        input_174 = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_ = (None)
        input_176 = torch._C._nn.gelu(input_175, approximate="none")
        input_175 = None
        input_177 = torch._C._nn.linear(
            input_176,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_,
        )
        input_176 = l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_ = (None)
        input_178 = torch.permute(input_177, [0, 3, 1, 2])
        input_177 = None
        result_48 = (
            l_self_modules_features_modules_5_modules_18_parameters_layer_scale_
            * input_178
        )
        l_self_modules_features_modules_5_modules_18_parameters_layer_scale_ = (
            input_178
        ) = None
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        result_48 += result_47
        result_49 = result_48
        result_48 = result_47 = None
        input_179 = torch.conv2d(
            result_49,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_ = (None)
        input_180 = torch.permute(input_179, [0, 2, 3, 1])
        input_179 = None
        input_181 = torch.nn.functional.layer_norm(
            input_180,
            (512,),
            l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_180 = l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_ = (None)
        input_182 = torch._C._nn.linear(
            input_181,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_,
        )
        input_181 = l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_ = (None)
        input_183 = torch._C._nn.gelu(input_182, approximate="none")
        input_182 = None
        input_184 = torch._C._nn.linear(
            input_183,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_,
        )
        input_183 = l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_ = (None)
        input_185 = torch.permute(input_184, [0, 3, 1, 2])
        input_184 = None
        result_50 = (
            l_self_modules_features_modules_5_modules_19_parameters_layer_scale_
            * input_185
        )
        l_self_modules_features_modules_5_modules_19_parameters_layer_scale_ = (
            input_185
        ) = None
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        result_50 += result_49
        result_51 = result_50
        result_50 = result_49 = None
        input_186 = torch.conv2d(
            result_51,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_ = (None)
        input_187 = torch.permute(input_186, [0, 2, 3, 1])
        input_186 = None
        input_188 = torch.nn.functional.layer_norm(
            input_187,
            (512,),
            l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_187 = l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_ = (None)
        input_189 = torch._C._nn.linear(
            input_188,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_,
        )
        input_188 = l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_ = (None)
        input_190 = torch._C._nn.gelu(input_189, approximate="none")
        input_189 = None
        input_191 = torch._C._nn.linear(
            input_190,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_,
        )
        input_190 = l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_ = (None)
        input_192 = torch.permute(input_191, [0, 3, 1, 2])
        input_191 = None
        result_52 = (
            l_self_modules_features_modules_5_modules_20_parameters_layer_scale_
            * input_192
        )
        l_self_modules_features_modules_5_modules_20_parameters_layer_scale_ = (
            input_192
        ) = None
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        result_52 += result_51
        result_53 = result_52
        result_52 = result_51 = None
        input_193 = torch.conv2d(
            result_53,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_ = (None)
        input_194 = torch.permute(input_193, [0, 2, 3, 1])
        input_193 = None
        input_195 = torch.nn.functional.layer_norm(
            input_194,
            (512,),
            l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_194 = l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_ = (None)
        input_196 = torch._C._nn.linear(
            input_195,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_,
        )
        input_195 = l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_ = (None)
        input_197 = torch._C._nn.gelu(input_196, approximate="none")
        input_196 = None
        input_198 = torch._C._nn.linear(
            input_197,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_,
        )
        input_197 = l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_ = (None)
        input_199 = torch.permute(input_198, [0, 3, 1, 2])
        input_198 = None
        result_54 = (
            l_self_modules_features_modules_5_modules_21_parameters_layer_scale_
            * input_199
        )
        l_self_modules_features_modules_5_modules_21_parameters_layer_scale_ = (
            input_199
        ) = None
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        result_54 += result_53
        result_55 = result_54
        result_54 = result_53 = None
        input_200 = torch.conv2d(
            result_55,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_ = (None)
        input_201 = torch.permute(input_200, [0, 2, 3, 1])
        input_200 = None
        input_202 = torch.nn.functional.layer_norm(
            input_201,
            (512,),
            l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_201 = l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_ = (None)
        input_203 = torch._C._nn.linear(
            input_202,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_,
        )
        input_202 = l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_ = (None)
        input_204 = torch._C._nn.gelu(input_203, approximate="none")
        input_203 = None
        input_205 = torch._C._nn.linear(
            input_204,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_,
        )
        input_204 = l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_ = (None)
        input_206 = torch.permute(input_205, [0, 3, 1, 2])
        input_205 = None
        result_56 = (
            l_self_modules_features_modules_5_modules_22_parameters_layer_scale_
            * input_206
        )
        l_self_modules_features_modules_5_modules_22_parameters_layer_scale_ = (
            input_206
        ) = None
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        result_56 += result_55
        result_57 = result_56
        result_56 = result_55 = None
        input_207 = torch.conv2d(
            result_57,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_ = (None)
        input_208 = torch.permute(input_207, [0, 2, 3, 1])
        input_207 = None
        input_209 = torch.nn.functional.layer_norm(
            input_208,
            (512,),
            l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_208 = l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_ = (None)
        input_210 = torch._C._nn.linear(
            input_209,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_,
        )
        input_209 = l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_ = (None)
        input_211 = torch._C._nn.gelu(input_210, approximate="none")
        input_210 = None
        input_212 = torch._C._nn.linear(
            input_211,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_,
        )
        input_211 = l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_ = (None)
        input_213 = torch.permute(input_212, [0, 3, 1, 2])
        input_212 = None
        result_58 = (
            l_self_modules_features_modules_5_modules_23_parameters_layer_scale_
            * input_213
        )
        l_self_modules_features_modules_5_modules_23_parameters_layer_scale_ = (
            input_213
        ) = None
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        result_58 += result_57
        result_59 = result_58
        result_58 = result_57 = None
        input_214 = torch.conv2d(
            result_59,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_ = (None)
        input_215 = torch.permute(input_214, [0, 2, 3, 1])
        input_214 = None
        input_216 = torch.nn.functional.layer_norm(
            input_215,
            (512,),
            l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_215 = l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_ = (None)
        input_217 = torch._C._nn.linear(
            input_216,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_,
        )
        input_216 = l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_ = (None)
        input_218 = torch._C._nn.gelu(input_217, approximate="none")
        input_217 = None
        input_219 = torch._C._nn.linear(
            input_218,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_,
        )
        input_218 = l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_ = (None)
        input_220 = torch.permute(input_219, [0, 3, 1, 2])
        input_219 = None
        result_60 = (
            l_self_modules_features_modules_5_modules_24_parameters_layer_scale_
            * input_220
        )
        l_self_modules_features_modules_5_modules_24_parameters_layer_scale_ = (
            input_220
        ) = None
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        result_60 += result_59
        result_61 = result_60
        result_60 = result_59 = None
        input_221 = torch.conv2d(
            result_61,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_ = (None)
        input_222 = torch.permute(input_221, [0, 2, 3, 1])
        input_221 = None
        input_223 = torch.nn.functional.layer_norm(
            input_222,
            (512,),
            l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_222 = l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_ = (None)
        input_224 = torch._C._nn.linear(
            input_223,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_,
        )
        input_223 = l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_ = (None)
        input_225 = torch._C._nn.gelu(input_224, approximate="none")
        input_224 = None
        input_226 = torch._C._nn.linear(
            input_225,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_,
        )
        input_225 = l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_ = (None)
        input_227 = torch.permute(input_226, [0, 3, 1, 2])
        input_226 = None
        result_62 = (
            l_self_modules_features_modules_5_modules_25_parameters_layer_scale_
            * input_227
        )
        l_self_modules_features_modules_5_modules_25_parameters_layer_scale_ = (
            input_227
        ) = None
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        result_62 += result_61
        result_63 = result_62
        result_62 = result_61 = None
        input_228 = torch.conv2d(
            result_63,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_ = (None)
        input_229 = torch.permute(input_228, [0, 2, 3, 1])
        input_228 = None
        input_230 = torch.nn.functional.layer_norm(
            input_229,
            (512,),
            l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_229 = l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_ = (None)
        input_231 = torch._C._nn.linear(
            input_230,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_,
        )
        input_230 = l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_ = (None)
        input_232 = torch._C._nn.gelu(input_231, approximate="none")
        input_231 = None
        input_233 = torch._C._nn.linear(
            input_232,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_,
        )
        input_232 = l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_ = (None)
        input_234 = torch.permute(input_233, [0, 3, 1, 2])
        input_233 = None
        result_64 = (
            l_self_modules_features_modules_5_modules_26_parameters_layer_scale_
            * input_234
        )
        l_self_modules_features_modules_5_modules_26_parameters_layer_scale_ = (
            input_234
        ) = None
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        result_64 += result_63
        result_65 = result_64
        result_64 = result_63 = None
        x_9 = result_65.permute(0, 2, 3, 1)
        result_65 = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (512,),
            l_self_modules_features_modules_6_modules_0_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_parameters_bias_,
            1e-06,
        )
        x_9 = (
            l_self_modules_features_modules_6_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_0_parameters_bias_ = None
        x_11 = x_10.permute(0, 3, 1, 2)
        x_10 = None
        input_235 = torch.conv2d(
            x_11,
            l_self_modules_features_modules_6_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = (
            l_self_modules_features_modules_6_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_1_parameters_bias_ = None
        input_236 = torch.conv2d(
            input_235,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_237 = torch.permute(input_236, [0, 2, 3, 1])
        input_236 = None
        input_238 = torch.nn.functional.layer_norm(
            input_237,
            (1024,),
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_237 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_239 = torch._C._nn.linear(
            input_238,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_238 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_240 = torch._C._nn.gelu(input_239, approximate="none")
        input_239 = None
        input_241 = torch._C._nn.linear(
            input_240,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_240 = l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_242 = torch.permute(input_241, [0, 3, 1, 2])
        input_241 = None
        result_66 = (
            l_self_modules_features_modules_7_modules_0_parameters_layer_scale_
            * input_242
        )
        l_self_modules_features_modules_7_modules_0_parameters_layer_scale_ = (
            input_242
        ) = None
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        result_66 += input_235
        result_67 = result_66
        result_66 = input_235 = None
        input_243 = torch.conv2d(
            result_67,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_244 = torch.permute(input_243, [0, 2, 3, 1])
        input_243 = None
        input_245 = torch.nn.functional.layer_norm(
            input_244,
            (1024,),
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_244 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_246 = torch._C._nn.linear(
            input_245,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_245 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_247 = torch._C._nn.gelu(input_246, approximate="none")
        input_246 = None
        input_248 = torch._C._nn.linear(
            input_247,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_247 = l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_249 = torch.permute(input_248, [0, 3, 1, 2])
        input_248 = None
        result_68 = (
            l_self_modules_features_modules_7_modules_1_parameters_layer_scale_
            * input_249
        )
        l_self_modules_features_modules_7_modules_1_parameters_layer_scale_ = (
            input_249
        ) = None
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        result_68 += result_67
        result_69 = result_68
        result_68 = result_67 = None
        input_250 = torch.conv2d(
            result_69,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_251 = torch.permute(input_250, [0, 2, 3, 1])
        input_250 = None
        input_252 = torch.nn.functional.layer_norm(
            input_251,
            (1024,),
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_251 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_253 = torch._C._nn.linear(
            input_252,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_252 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_254 = torch._C._nn.gelu(input_253, approximate="none")
        input_253 = None
        input_255 = torch._C._nn.linear(
            input_254,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_254 = l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_256 = torch.permute(input_255, [0, 3, 1, 2])
        input_255 = None
        result_70 = (
            l_self_modules_features_modules_7_modules_2_parameters_layer_scale_
            * input_256
        )
        l_self_modules_features_modules_7_modules_2_parameters_layer_scale_ = (
            input_256
        ) = None
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        result_70 += result_69
        result_71 = result_70
        result_70 = result_69 = None
        x_12 = torch.nn.functional.adaptive_avg_pool2d(result_71, 1)
        result_71 = None
        x_13 = x_12.permute(0, 2, 3, 1)
        x_12 = None
        x_14 = torch.nn.functional.layer_norm(
            x_13,
            (1024,),
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
            1e-06,
        )
        x_13 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        x_15 = x_14.permute(0, 3, 1, 2)
        x_14 = None
        input_257 = x_15.flatten(1, -1)
        x_15 = None
        input_258 = torch._C._nn.linear(
            input_257,
            l_self_modules_classifier_modules_2_parameters_weight_,
            l_self_modules_classifier_modules_2_parameters_bias_,
        )
        input_257 = (
            l_self_modules_classifier_modules_2_parameters_weight_
        ) = l_self_modules_classifier_modules_2_parameters_bias_ = None
        return (input_258,)
