# config.py

class Hyperparams:
    # 数据参数
    pkl_path_04 = 'pems04_60m.pkl'
    pkl_path_08 = 'pems08_60m.pkl'
    input_max_len = 12
    output_max_len = 12
    # 空洞卷积配置
    cnn_dilation_rates = [1, 2, 4, 8]  # 对应kernel_sizes的数量
    # 模型参数
    maxlen = input_max_len
    outputlen = output_max_len
    input_size = 1
    #hidden_size = 128  # 也用作 d_model
    #hidden_size = 16  # 也用作 d_model
    hidden_size = 8
    hidden_size16 = 16
    num_layers = 2
    output_size = 1
    residual_connections = 0  # 示例：设置为2，可以根据需要调整为0到5之间的任何值

    # 新增Informer特有参数
    enc_layers = 3          # 编码器层数
    attn_sample_factor = 5 # 注意力采样因子

    # Transformer 特有的参数
    kernel_sizes_per_head = [[1, 3], [3, 5], [5, 7], [7, 9], [1, 5], [3, 7], [5, 9], [1, 7]]  # 为每个头指定不同的卷积核尺寸
    nhead = 8  # 头的数量改为8

    nhead16 = 16  # 头的数量更新为16
    kernel_sizes_per_head16 = [
        [1, 3], [3, 5], [5, 7], [7, 9],
        [1, 5], [3, 7], [5, 9], [1, 7],
        [1, 9], [2, 6], [4, 8], [3, 9],
        [2, 4], [4, 6], [6, 8], [8, 10]  # 为16个头指定不同的卷积核尺寸
    ]

    head_configs = [
                       # (kernel_sizes, dilation_rates)
                       ([3, 5], [1, 2]),  # 头1：小核+小空洞
                       ([5, 7], [2, 3]),  # 头2：中核+中空洞
                       ([7, 9], [3, 4]),  # 头3：大核+大空洞
                       ([3, 7], [1, 3]),  # 头4：混合尺寸
                       ([5, 9], [2, 4]),  # 头5：大核组合
                       ([3, 5, 7], [1, 2, 3]),  # 头6：多尺度组合
                       ([9, 11], [4, 5]),  # 头7：超大核+大空洞（捕获长程模式）
                       ([3, 5, 7, 9], [1, 2, 3, 4])  # 头8：多尺度密集组合
                   ][:nhead]  # 根据实际头数自动截取

    nhead4 = 4  # 头的数量更新为16
    kernel_sizes_per_head4 = [
        [1, 3], [3, 5], [5, 7], [7, 9]
    ]

    head_out_channels = 64  # 每个头的输出通道数
    dim_feedforward = 512
    dropout_rate = 0.1
    activation_function = 'relu'
    emb_size = 128
    max_position_embeddings = 5000
    patch_size = 16  # 例如，对于序列数据，这需要根据你的数据调整
    mlp_dim = 512  # Transformer内部前馈网络的维度
    num_transformer_layers = 6  # Transformer编码器层的数量

    # CNN 特有的参数
    num_cnn_layers = 4  # 根据 kernel_sizes_per_head 的长度设置
    kernel_sizes = [3, 5, 7, 9]  # 默认卷积核大小，如果使用多尺度，这个参数将被忽略

    kernel_size = 3
    out_channels = 64  # 卷积层的输出通道数
    cnn_dropout_rate = 0.3  # CNN 层的 dropout 比率
    inception_output_size = 4 * out_channels  # 因为有四个分支
    weight_decay = 1e-4
    growth_rate = 32  # 新增的参数

    # 训练参数
    batch_size = 128
    num_epochs = 100
    #num_epochs = 1 #调图用
    learning_rate = 0.001

    # 其他参数
    sequence_length = 288
    future_seq = 12
