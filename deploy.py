import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms


MODEL_PATH = "animal_classifier_resnet50.pth"
CLASS_NAMES = ['Peacock', 'bonnet_macaque', 'chital', 'elephant', 'no animal', 'pig', 'porcupine', 'street_dogs',
               'wild_boar']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(2)

print("Starting live test. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break

    input_tensor = preprocess(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

    label = f"{CLASS_NAMES[preds[0]]} ({confidence:.2f})"

    color = (0, 255, 0) if CLASS_NAMES[preds[0]] != 'no animal' else (0, 0, 255)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('AI Scarecrow - Laptop Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()