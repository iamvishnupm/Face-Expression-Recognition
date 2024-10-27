from django.shortcuts import render
import json
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import numpy as np
import cv2
from .video_prediction import video_prediction


@csrf_exempt
def process_frame(request):
    if request.method == "POST":
        try:
            # Use request.body and json.loads to parse incoming JSON
            data = json.loads(request.body)
            image_data = data.get("image")

            if not image_data:
                return JsonResponse({"error": "No image data provided"}, status=400)

            # Decode the base64 image
            image_data = image_data.split(",")[1]  # Remove the base64 header
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert PIL image to OpenCV format (numpy array)
            image_np = np.array(image)

            # Flip the image horizontally (OpenCV)
            flipped_image = cv2.flip(image_np, 1)

            img_res = video_prediction(flipped_image)

            # Convert OpenCV image back to PIL
            img_res_pil = Image.fromarray(img_res)

            # Convert PIL image to base64 for sending back to client
            buffer = io.BytesIO()
            img_res_pil.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Return the processed image
            return JsonResponse({"processed_image": encoded_image})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


def home(request):
    return render(request, "home.html", {})


def introduction(request):
    return render(request, "introduction.html", {})


def training(request):
    return render(request, "training.html", {})


def prediction(request):
    return render(request, "prediction.html", {})


def live(request):
    return render(request, "live.html", {})
