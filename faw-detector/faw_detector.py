#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to run generic MobileNet based classification model."""
import argparse
import time
import queue
import threading


from picamera import Color
from picamera import PiCamera

from aiy.vision import inference
from aiy.vision.models import utils

#import libraries for tone generator
from aiy.toneplayer import TonePlayer 

#import send message from locall python file fona
from ./fona import send_message

class Service(object):

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is None:
                break
            self.process(request)
            self._requests.task_done()

    def join(self):
        self._thread.join()

    def stop(self):
        self._requests.put(None)

    def process(self, request):
        pass

    def submit(self, request):
        self._requests.put(request)



class Player(Service):
    """Controls buzzer."""

  def __init__(self, gpio, bpm):
    super().__init__()
    self._toneplayer = TonePlayer(gpio, bpm)

  def process(self, sound):
    self._toneplayer.play(*sound)

  def play(self, sound):
    self.submit(sound)

def read_labels(label_path):
  with open(label_path) as label_file:
    return [label.strip() for label in label_file.readlines()]


def get_message(processed_result, threshold, top_k):
  if processed_result:
    message = 'Detecting:\n %s' % ('\n'.join(processed_result))
  else:
    message = 'Nothing detected when threshold=%.2f, top_k=%d' % (
              threshold, top_k)
  return message

def process(result, labels, out_tensor_name, threshold, top_k):
  """Processes inference result and returns labels sorted by confidence."""
  # MobileNet based classification model returns one result vector.
  assert len(result.tensors) == 1
  tensor = result.tensors[out_tensor_name]
  probs, shape = tensor.data, tensor.shape
  assert shape.depth == len(labels)
  pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
  pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
  pairs = pairs[0:top_k]
  return [' %s (%.2f)' % (labels[index], prob) for index, prob in pairs]



def detection_made(processed_result, detection_logger):
  if processed_result in args.hunting && detection_logger[processed_result] < 3:
    detection_logger[processed_result] += 1
  else if detection_logger[processed_result] = args.message_threshold:
    detection_logger[processed_result] = 0
    send_message(processed_result)
    #make noise
    player.play('C5q', 'E5q', 'C6q')
  else:
    return 

detection_logger = {}




def main():
#define player for making noises
  player=Player(gpio=22,bpm=10)
   

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',
      required=True,
      help='Path to converted model file that can run on VisionKit.')
  parser.add_argument(
      '--label_path',
      required=True,
      help='Path to label file that corresponds to the model.')
  parser.add_argument(
      '--input_height', type=int, required=True, help='Input height.')
  parser.add_argument(
      '--input_width', type=int, required=True, help='Input width.')
  parser.add_argument(
      '--input_layer', required=True, help='Name of input layer.')
  parser.add_argument(
      '--output_layer', required=True, help='Name of output layer.')
  parser.add_argument(
      '--num_frames',
      type=int,
      default=-1,
      help='Sets the number of frames to run for, otherwise runs forever.')
  parser.add_argument(
      '--input_mean', type=float, default=128.0, help='Input mean.')
  parser.add_argument(
      '--input_std', type=float, default=128.0, help='Input std.')
  parser.add_argument(
      '--input_depth', type=int, default=3, help='Input depth.')
  parser.add_argument(
      '--threshold', type=float, default=0.1,
      help='Threshold for classification score (from output tensor).')
  parser.add_argument(
      '--top_k', type=int, default=3, help='Keep at most top_k labels.')
  parser.add_argument(
      '--preview',
      action='store_true',
      default=False,
      help='Enables camera preview in addition to printing result to terminal.')
  parser.add_argument(
      '--show_fps',
      action='store_true',
      default=False,
      help='Shows end to end FPS.')

  parser.add_argument(
      '--detecting_list',
      type=list,
      default=[],
      help='Input a list of bugs that you want to keep.'
  )
  parser.add_argument{
    '--message_threshold',type=int,default=3,help='Input detection threshold for sending sms'
  )
  args = parser.parse_args()

  for item in args.detecting_list:
    detection_logger.item = 0 
  	

  model = inference.ModelDescriptor(
      name='mobilenet_based_classifier',
      input_shape=(1, args.input_height, args.input_width, args.input_depth),
      input_normalizer=(args.input_mean, args.input_std),
      compute_graph=utils.load_compute_graph(args.model_path))
  labels = read_labels(args.label_path)

  with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
    if args.preview:
      camera.start_preview()
    with inference.CameraInference(model) as camera_inference:
      last_time = time.time()
      for i, result in enumerate(camera_inference.run()):
        if i == args.num_frames:
          break
        processed_result = process(result, labels, args.output_layer,
                                   args.threshold, args.top_k)
        #my function to handle sending messages if detection happens at the threshold.
        detection_made(processed_result)
        cur_time = time.time()
        fps = 1.0 / (cur_time - last_time)
        last_time = cur_time

        message = get_message(processed_result, args.threshold, args.top_k)
		
        if args.show_fps:
          message += '\nWith %.1f FPS.' % fps


        print(message)

        if args.preview:
          camera.annotate_foreground = Color('black')
          camera.annotate_background = Color('white')
          # PiCamera text annotation only supports ascii.
          camera.annotate_text = '\n %s' % message.encode(
              'ascii','backslashreplace').decode('ascii')

    if args.preview:
      camera.stop_preview()

if __name__ == '__main__':
  main()
