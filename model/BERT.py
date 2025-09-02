import torch
import torch.nn as nn

class BERT(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(BERT, self).__init__()
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出维度为2，表示二分类
            nn.Linear(d_model, 2)
        )
        
    def forward(self, x, mask=None):
        # x shape: (seq_len, batch_size, d_model)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 取序列的第一个token的表示作为整个序列的表示
        sequence_output = encoded[0, :, :]  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(sequence_output)  # (batch_size, 2)
        return logits
        
def train_example():
    # 创建模型
    model = BERT()
    
    # 假设我们有一个批次的数据
    batch_size = 32
    seq_length = 50
    d_model = 768
    
    # 创建随机输入数据
    x = torch.randn(seq_length, batch_size, d_model)  # (seq_len, batch_size, d_model)
    labels = torch.randint(0, 2, (batch_size,))  # (batch_size,)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    logits = model(x)
    loss = criterion(logits, labels)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 测试模式
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(seq_length, batch_size, d_model)
        predictions = model(test_x)
        predicted_labels = torch.argmax(predictions, dim=1)

if __name__ == "__main__":
    train_example()
