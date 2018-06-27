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
import signal
import logging

from picamera import Color
from picamera import PiCamera

from aiy.vision import inference
from aiy.vision.models import utils

from gpiozero import Button

from aiy.leds import Leds
#import libraries for tone generator
from aiy.toneplayer import TonePlayer

import aiy._drivers._button
leds = Leds()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#import send message from locall python file fona
#from ./fona import send_message


GREEN = (0x00,0xFF,0x00)

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')

  

leds = Leds()
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

def read_labels(label_path):
    with open(label_path) as label_file:
        return [label.strip() for label in label_file.readlines()]


def get_message(processed_result, threshold, top_k):
    if processed_result:
        message = 'Detecting:\n %s' % ('\n'.join(processed_result))
    else:
        message = 'Nothing detected when threshold=%.2f, top_k=%d' % (threshold, top_k)
    return message


def detection_made(processed_result, detection_logger, message_threshold, detecting_list):
    for bug in processed_result:
        logger.info(bug)
        if bug in detecting_list:
            if detection_logger[bug] == message_threshold:
                logger.info('hasnt met threshold')
                detection_logger[bug] += 1
            elif detection_logger[bug] < message_threshold:
                logger.info(bug)
                detection_logger[bug] = 0
                #    send_message(processed_result)
                #make noise
                player.play(BEEP_SOUND)
                return
        else:
            return

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



class FawDetector(Service):
    def __init__(self):
        self._done = threading.Event()
        signal.signal(signal.SIGINT, lambda signal, frame: self.stop())
        signal.signal(signal.SIGTERM, lambda signal, frame: self.stop())

    def stop(self):
        logger.info('Stopping...')
        self._done.set()

    def run(self,input_layer,output_layer,num_frames, input_mean, input_std, threshold, top_k, detecting_list,message_threshold, model,labels):
        detection_logger = {}
        for item in detecting_list:
            detection_logger.update({item:0})

        #logging.info(detection_logger)
        logger.info('Starting...')
        player = Player(gpio=22, bpm=10)
        try:
            with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
                with inference.CameraInference(model) as camera_inference:
                    last_time = time.time()
                    logger.info('Model loaded.')
                    player.play(MODEL_LOAD_SOUND)
                    for i, result in enumerate(camera_inference.run()):
                        if i == num_frames or self._done.is_set():
                            break
                        processed_result = process(result, labels, output_layer,threshold, top_k)
                        #logger.info('Processed result')

            #my function to handle sending messages if detection happens at the threshold.
                        detection_made(processed_result, detection_logger, message_threshold, detecting_list)
                        cur_time = time.time()
                        fps = 1.0 / (cur_time - last_time)
                        last_time = cur_time
                        message = get_message(processed_result, threshold, top_k)
                       # logger.info(message)
        finally:
            player.stop()
            player.join()

def main():



    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--input_layer',  default='map/TensorArrayStack/TensorArrayGatherV3', help='Name of input layer.')
    parser.add_argument(
            '--output_layer',  default="prediction", help='Name of output layer.')
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
        '--threshold', type=float, default=0.1,help='Threshold for classification score (from output tensor).')
    parser.add_argument(
        '--top_k', type=int, default=3, help='Keep at most top_k labels.')
    parser.add_argument(
        '--detecting_list',
        type=list,
        default=['Biston betularia (Peppered Moth)','Spodoptera litura (Oriental Leafworm Moth)','Utetheisa ornatrix (Ornate Bella Moth)','Polygrammate hebraeicum (Hebrew Moth)','Palpita magniferalis (Splendid Palpita Moth) (0.14)'],
        help='Input a list of bugs that you want to keep.')
    parser.add_argument(
        '--message_threshold',type=int,default=1,help='Input detection threshold for sending sms'
        )
    args = parser.parse_args()
    model = inference.ModelDescriptor(
        name='mobilenet_based_classifier',
        input_shape=(1, 192, 192, 3),
        input_normalizer=(128.0, 128.0),
        compute_graph=utils.load_compute_graph('mobilenet_v2_192res_1.0_inat_insect.binaryproto'))
    labels = read_labels("/home/pi/models/mobilenet_v2_192res_1.0_inat_insect_labels.txt")
    detector = FawDetector()

    detector.run(args.input_layer,args.output_layer, args.num_frames, args.input_mean, args.input_std, args.threshold, args.top_k, args.detecting_list, args.message_threshold, model, labels)

if __name__ == '__main__':
    main()
