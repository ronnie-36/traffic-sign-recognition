import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from app.torch_utils import AlexnetTS
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

torch.device("cpu")
# Modelling Task
model = AlexnetTS(43)
model.load_state_dict(torch.load("app/model.pth", map_location=torch.device("cpu")))
model.eval()

classes = {
    1: "Speed limit (20km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (50km/h)",
    4: "Speed limit (60km/h)",
    5: "Speed limit (70km/h)",
    6: "Speed limit (80km/h)",
    7: "End of speed limit (80km/h)",
    8: "Speed limit (100km/h)",
    9: "Speed limit (120km/h)",
    10: "No passing",
    11: "No passing veh over 3.5 tons",
    12: "Right-of-way at intersection",
    13: "Priority road",
    14: "Yield",
    15: "Stop",
    16: "No vehicles",
    17: "Veh > 3.5 tons prohibited",
    18: "No entry",
    19: "General caution",
    20: "Dangerous curve left",
    21: "Dangerous curve right",
    22: "Double curve",
    23: "Bumpy road",
    24: "Slippery road",
    25: "Road narrows on the right",
    26: "Road work",
    27: "Traffic signals",
    28: "Pedestrians",
    29: "Children crossing",
    30: "Bicycles crossing",
    31: "Beware of ice/snow",
    32: "Wild animals crossing",
    33: "End speed + passing limits",
    34: "Turn right ahead",
    35: "Turn left ahead",
    36: "Ahead only",
    37: "Go straight or right",
    38: "Go straight or left",
    39: "Keep right",
    40: "Keep left",
    41: "Roundabout mandatory",
    42: "End of no passing",
    43: "End no passing vehicle with a weight greater than 3.5 tons",
}

num = range(43)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize([112, 112]), transforms.ToTensor()]
    )

    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    y_test_pred = model(tensor)
    y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    y_pred_tags = y_pred_tags.cpu().numpy()

    y_pred = y_pred_tags[0]
    y_pred = labels[y_pred]
    return classes[y_pred + 1]


# Treat the web process
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.get("file")
        if not file:
            return
        img_bytes = file.read()
        prediction_name = get_prediction(img_bytes)
        return render_template(
            "result.html",
            name=prediction_name,
        )

    return render_template("index.html")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)
