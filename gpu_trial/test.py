import torch
print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(100, 100).to(device)
w = torch.randn(100, 100).to(device)
y = torch.matmul(x, w)
print(y.device)  # 應顯示 cuda:0