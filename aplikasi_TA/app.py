from flask import Flask, render_template, request
import tensorflow as tf
import keras
from keras.preprocessing import image
import cv2
import numpy as np
import random
import imageio
from tensorflow_docs.vis import embed
# import base64

app = Flask(__name__)

model = tf.keras.models.load_model('checkpoint_31')

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]

HEIGHT = 224
WIDTH = 224

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        video = request.files['my_video']

        video_path = "static/" + video.filename
        video.save(video_path)

        frames = frames_from_video_file(video_path, 10)

        gif_path = to_gif(frames, video.filename)

        prediction, tingkat_keyakinan = predict_labels(frames)

    return render_template("index.html", prediction=prediction, tingkat_keyakinan=tingkat_keyakinan, gif_path=gif_path)

def frames_from_video_file(video_path, n_frames, output_size = (HEIGHT,WIDTH)):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  # Frame step fleksibel mengikuti panjang dari sumber video
  frame_step = max(int(video_length / (n_frames-1)), 1)

  need_length = 1 + (n_frames-1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)

  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret and frame is not None:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))

  src.release()
  result = np.array(result)[..., [2, 1, 0]]
  return result


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

        Return:
          Formatted frame with padding of specified output size.
    """
    # return resized_frame
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def to_gif(images, video_name):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    file_path = (f'./gif/{video_name}.gif')
    save_path = "static/" + file_path
    imageio.mimsave(save_path, converted_images, fps=2)
    # return embed.embed_file(f'./gif/{video_name}.gif')
    return file_path

def predict_labels(frames):
    dic = {0 : 'Card', 1 : 'Celebration', 2 : 'Corner', 3 : 'Foul', 4 : 'Shot', 5 : 'Substitution'}

    expanded_frames = np.expand_dims(frames, axis=0)

    predictions = model.predict(expanded_frames)

    predicted_classes = np.argmax(predictions, axis=1)

    return dic[predicted_classes[0]], predictions[0][predicted_classes[0]]


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)