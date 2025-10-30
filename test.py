print("H1ello World")
import torch
import torch.nn as nn
import time
# 1. Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(in_features=1000, out_features=2000)
        self.layer2 = nn.Linear(in_features=2000, out_features=2000)
        self.layer3 = nn.Linear(in_features=2000, out_features=2000)
        self.layer4 = nn.Linear(in_features=2000, out_features=5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# 2. Check for GPU availability and set the device
# This line checks if an NVIDIA GPU is available and PyTorch can use it (CUDA).
device = torch.device("cuda")
print(f"Using device: {device} 💻")


# 3. Instantiate the model
model = SimpleModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
inputs = torch.randn(64, 1000).to(device)
labels = torch.randint(0, 5, (64,)).to(device)
model.train()
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
model.eval()
# 4. Move the model to the selected device (GPU or CPU)
# This is the key step. It moves all of the model's parameters and buffers
# to the GPU's memory.
torch.save(model.state_dict(),"test.pt")
print("\nModel successfully loaded onto the GPU.")
# You can verify the device of a model parameter
print(f"Device of layer1's weights: {next(model.parameters()).device}")


# --- Important Note ---
# When you want to perform calculations, your input data (tensors) must
# also be on the same device as the model.
print("\n--- Example Usage ---")
# Create a random input tensor on the CPU
input_tensor_cpu = torch.randn(320, 1000) # 32 samples, 10 features
print(f"Input tensor's initial device: {input_tensor_cpu.device}")

# Move the input tensor to the same device as the model
input_tensor_gpu = input_tensor_cpu.to(device)
print(f"Input tensor's device after moving: {input_tensor_gpu.device}")

# Now you can perform a forward pass
# The computation will happen on the GPU
output = model(input_tensor_gpu)
print(f"Output tensor's device: {output.device}")
time.sleep(20)
