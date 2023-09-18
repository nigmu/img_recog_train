# from flask import Flask, request, render_template
# from keras.models import load_model
# import cv2
# import numpy as np
# import tensorflow as tf
# from image_processing import preprocess
# from num_to_label import num_to_label
# import time
# import pandas as pd


# app = Flask(_name_)

# # Load your model here
# model = load_model(
#     "C:\\Users\\nigam\\Downloads\\saved_hdf5_model.h5", compile=False
# )  # Disable model compilation

# predicted_labels = []
# predicted_labels_timestamps = []


# @app.route("/", methods=["GET", "POST"])
# def upload_file():
#     global predicted_labels
#     global predicted_labels_timestamps

#     if request.method == "POST":
#         file = request.files["file"]
#         if not file:
#             return render_template(
#                 "index.html",
#                 prediction={"text": "No file found. Please upload an image."},
#             )

#         # Read the image in grayscale
#         image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

#         # Preprocess the image here
#         image = preprocess(image, (128, 32))
#         image = image / 255.0

#         # Predict the label for the image
#         pred = model.predict(image.reshape(1, 128, 32, 1))
#         decoded = tf.keras.backend.get_value(
#             tf.keras.backend.ctc_decode(
#                 pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True
#             )[0][0]
#         )

#         # Convert numerical predictions to labels
#         predicted_label = num_to_label(decoded[0])
#         print(type(predicted_label))

#         predicted_labels.append(predicted_label)

#         timestamp = time.strftime("%Y-%m-%D %H:%M:%S")
#         predicted_labels_timestamps.append(timestamp)
#         print(predicted_labels_timestamps)
#         print(predicted_labels)

#         # data = {"lader": predicted_labels, "time_date": predicted_labels_timestamps}
#         # ex = pandas.DataFrame(data)
#         # ex.to_csv("tracking.csv")

#         # Return the predicted label
#         return render_template(
#             "index.html",
#             prediction={"text": "Predicted Label: {}".format(predicted_label)},
#         )

#     return render_template("index.html", prediction={"text": ""})


# if _name_ == "_main_":
#     app.run(debug=True)


import os
import cv2
from flask import Flask, request, render_template,send_file
from keras.models import load_model
import numpy as np
import tensorflow as tf
from image_processing import preprocess
from num_to_label import num_to_label
import time
import tempfile
import pandas
import mysql.connector as con

app = Flask(__name__)

# Load your model here
model = load_model(
    "Team-71\\Backend\\model\\final_hdf5_model_3.h5", compile=False
)  # Disable model compilation

predicted_labels = []
predicted_labels_timestamps = []
arr2 = []


def extract_frames(video, interval):
    # Initialize frame counter
    frame_counter = 0

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # If the frame was not successfully read then we have reached the end of the video
        if not ret:
            break

        # If this is the right frame to save based on the interval (in seconds)
        if frame_counter % (30 * interval) == 0:
            # Resize the frame to 196x28 pixels and convert it to grayscale
            resized_frame = cv2.cvtColor(
                cv2.resize(frame, (196, 28)), cv2.COLOR_BGR2GRAY
            )

            # Preprocess the image here
            image = preprocess(resized_frame, (128, 32))
            image = image / 255.0

            # Predict the label for the image
            pred = model.predict(image.reshape(1, 128, 32, 1))
            decoded, _ = tf.keras.backend.ctc_decode(
                pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True
            )

            # Convert numerical predictions to labels
            predicted_label = num_to_label(tf.keras.backend.get_value(decoded[0])[0])
            print(type(predicted_label))

            predicted_labels.append(predicted_label)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            predicted_labels_timestamps.append(timestamp)

            arr2 = [n for n in predicted_labels if n != "65688"]
            print(arr2)
            arr2_timestamps = [
                predicted_labels_timestamps[i]
                for i in range(len(predicted_labels))
                if predicted_labels[i] != "65688"
            ]
            print(arr2_timestamps)

            data = {"ladle": arr2, "time_date": arr2_timestamps}

            Data = pandas.DataFrame(data)
            Data.to_csv("frame.csv")
            # , mode='a', header=False, index=False

            # mysql databasre handle
            db1 = con.connect(
                host="localhost", user="root", password="12345", database="ladleworker"
            )

            cur = db1.cursor()
            s = "INSERT INTO ladle VALUES(%s,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            arr = arr2
            for x in arr:
                t = [
                    (x,),
                ]
                cur.executemany(s, t)
                db1.commit()

        # Increment our frame counter
        frame_counter += 1

    # Release the video file
    video.release()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global predicted_labels
    global predicted_labels_timestamps
    global arr2

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template(
                "index.html",
                prediction={"text": "No file found. Please upload a video."},
            )

        # Save the uploaded video file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        temp_file.close()

        # Open the video file
        video = cv2.VideoCapture(temp_file.name)

        # Extract frames and make predictions
        extract_frames(video, 5)

        # Delete the temporary file
        os.unlink(temp_file.name)

        return render_template(
            "index.html",
            prediction={"text": "Predicted Labels: {}".format(predicted_labels)},
        )

    return render_template("index.html", prediction={"text": ""})

@app.route('/download_csv')
def download_csv():
    return send_file('frame.csv', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
