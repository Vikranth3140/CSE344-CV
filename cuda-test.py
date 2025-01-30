import torch

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

# Get device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# Move a tensor to GPU
x = torch.rand(3, 3).to(device)
print("Tensor Device:", x.device)