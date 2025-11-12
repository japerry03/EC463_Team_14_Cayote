import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from mini_YOLO import MiniYOLO, decode_predictions, nms, letterbox

# ---------------------------
# Device Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ---------------------------
# Load YOLO Model
# ---------------------------
yolo_model = MiniYOLO(num_classes=2).to(device)
yolo_model.load_state_dict(torch.load("mini_yolo_test.pth", map_location=device))
yolo_model.eval()

# ---------------------------
# Actor-Critic Model
# ---------------------------
class CNNActorCritic(nn.Module):
    def __init__(self, action_dim=6):
        super().__init__()
        # Vision encoder (process 64x64 images)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # Compute CNN output dimension
        test_in = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            flat_dim = self.cnn(test_in).shape[1]
        # Fully connected layers with extra sensor inputs
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Actor and Critic heads
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, image, sensor):
        x = self.cnn(image)
        h = torch.cat([x, sensor], dim=1)
        h = self.fc(h)
        return self.actor(h), self.critic(h)

ac_model = CNNActorCritic(action_dim=6).to(device)
ac_model.eval()

# ---------------------------
# Stress Test Function
# ---------------------------
def stress_test(duration=15):
    start = time.time()
    frames = 0
    while time.time() - start < duration:
        # Simulated camera input (random image)
        frame = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
        img, ratio, pad = letterbox(frame, 640)
        x = torch.from_numpy(np.transpose(img.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)

        # YOLO Inference
        with torch.no_grad():
            preds = yolo_model(x)
            dets = decode_predictions(preds, conf_thresh=0.25, img_size=640)
            dets = nms(dets[0], iou_threshold=0.45)

        # Resize for Actor-Critic
        small_img = cv2.resize(frame, (64, 64))
        small_img = torch.from_numpy(np.transpose(small_img.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)

        # Simulated sensor input
        sensor = torch.randn(1, 16).to(device)

        # Actor-Critic Inference
        with torch.no_grad():
            actions, values = ac_model(small_img, sensor)

        frames += 1

    fps = frames / duration
    print(f"Average processing rate: {fps:.2f} FPS")

# ---------------------------
# Run the Stress Test
# ---------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance
    stress_test(20)
