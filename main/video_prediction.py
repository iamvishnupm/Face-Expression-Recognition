import os
import cv2
import torch
from PIL import Image
from .effnet import CustomModel
from torchvision import transforms
from django.contrib.staticfiles.storage import staticfiles_storage
from django.conf import settings

hcf_path = os.path.join(settings.MEDIA_ROOT, "haarcascade_frontalface_default.xml")
model_path = os.path.join(settings.MEDIA_ROOT, "model_state_dict.pth")


def video_prediction(frame):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = cv2.CascadeClassifier(hcf_path)

    model = CustomModel(7)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    faces = detector.detectMultiScale(frame, 1.1)

    for x, y, w, h in faces:
        face = frame[y - 10 : y + h + 10, x - 10 : x + w + 10]

        img = Image.fromarray(face)
        img = transform(img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            out = model(img)
            _, pred = torch.max(out, 1)

        pred = pred.item()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            frame,
            labels[pred],
            (x + w // 4, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

    return frame
