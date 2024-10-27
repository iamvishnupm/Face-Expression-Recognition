from django.urls import path
from .views import *

urlpatterns = [
    path("", home, name="home"),
    path("introduction/", introduction, name="introduction"),
    path("training/", training, name="training"),
    path("prediction/", prediction, name="prediction"),
    path("live/", live, name="live"),
    path("process_frame/", process_frame, name="process_frame"),
]
