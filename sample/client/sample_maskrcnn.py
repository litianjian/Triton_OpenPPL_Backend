# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:36:57 2022

@author: xusi
"""

import sys
import logging
import argparse
import numpy as np
import cv2

import tritonclient.grpc as grpcclient


logging.basicConfig(level=logging.INFO)

coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

class DataProcess(object):
    def __init__(self):
        self._w_scale = 1
        self._h_scale = 1
        self._mean = [123.675, 116.28, 103.53]
        self._std = [58.395, 57.12, 57.375]
        self._data_bin = 'input.bin'
        self._score_thr = 0.7

    def _load_and_resize(self, input_img_path, resized_h, resized_w):
        """Load an image from the specified input path, and resize it to requited shape(h, w)

        Keyword arguments:
        input_img_path -- string path of the image to be loaded
        resized_h -- required shape's h
        resized w -- required shape's w
        """
        self._input_file_name = input_img_path
        img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        self._h_scale = resized_h / img.shape[0]
        self._w_scale = resized_w / img.shape[1]
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        return img

    def _normalize(self, img):
        """ change input img data as a NumPy float array.

        Keyword arguments:
        img -- img's pixel array(numpy array)
        """
        mean = np.array(self._mean).reshape(1, -1).astype(np.float64)
        std = np.array(self._std).reshape(1, -1).astype(np.float64)
        stdinv = 1 / std
        img = img.copy().astype(np.float32)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        img = img.transpose(2, 0, 1)
        img.tofile(self._data_bin)

    def preprocess(self, input_img_path, resized_h, resized_w):
        """ change input img as a NumPy float array.

        Keyword arguments:
        input_img_path -- string path of the image to be loaded
        resized_h -- required shape's h
        resized w -- required shape's w
        """
        img = self._load_and_resize(input_img_path, resized_h, resized_w)
        self._normalize(img)
        return self._data_bin

    def _draw_result_to_img(self,
                            boxes_and_score_data,
                            labels_data,
                            masks_data,
                            save_file_name):
        """Draw the bounding boxes and mask on the original input image and save it.

        Keyword arguments:
        boxes_and_score_data -- NumPy array containing the bounding box coordinates of N objects and score, with shape (N,5).
        lables_data -- NumPy array containing the corresponding label for each object, with shape (N,)
        mask_data -- Numpy array containing the mask of N objects
        save_file_name -- out image file name
        """

        im = cv2.imread(self._input_file_name, cv2.IMREAD_COLOR)
        scores = boxes_and_score_data[:, -1]
        inds = scores > self._score_thr
        bboxes = boxes_and_score_data[inds, :]
        labels = labels_data[inds]
        segms = masks_data[inds, ...]

        np.random.seed(42)
        mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox_int = bbox.astype(np.int32)
            left = int(bbox_int[0] / self._w_scale)
            top = int(bbox_int[1] / self._h_scale)
            right = int(bbox_int[2] / self._w_scale)
            bottom = int(bbox_int[3] / self._h_scale)
            cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.putText(im,
                        coco_classes[label] + ": " + str(round(bbox[4], 2)),
                        (left, top),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1.2,
                        (0, 0, 0))
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i]
                mask = cv2.resize(mask, (im.shape[1], im.shape[0]))
                mask = mask.astype(bool)
                im[mask] = im[mask] * 0.5 + color_mask * 0.7
        cv2.imwrite(save_file_name, im)

    def postprocess(self, boxes_and_score_data, labels_data, masks_data, out_file_name):
        self._draw_result_to_img(boxes_and_score_data, labels_data, masks_data, out_file_name)


def parsArgs():
    parser = argparse.ArgumentParser("runner of maskrcnn onnx model.")
    parser.add_argument('-i', 
                        '--in_img_file', 
                        type=str, 
                        dest='in_img_file', 
                        required=False, 
                        default="test.jpg", 
                        help="Specify the input image.")
    parser.add_argument('-o', 
                        '--out_img_file', 
                        type=str, 
                        dest='out_img_file', 
                        required=False, 
                        default="test_out.jpg", 
                        help="Specify the output image's name.")

    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')
    parser.add_argument(
                        '-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates. Default is None.')
    parser.add_argument(
                        '-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key. Default is None.')
    parser.add_argument(
                        '-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain. Default is None.')
    parser.add_argument(
                        '-C',
                        '--grpc-compression-algorithm',
                        type=str,
                        required=False,
                        default=None,
                        help='The compression algorithm to be used when sending request to server. Default is None.')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsArgs()

    # prepare data
    image_file = args.in_img_file
    result_file = args.out_img_file

    data_process = DataProcess()
    # preprocess
    input0_data = data_process.preprocess(image_file, 800, 1200)
    input0_data = np.fromfile(input0_data, dtype=np.float32)

    # Setup triton client
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=args.url,
            verbose=args.verbose,
            ssl=args.ssl,
            root_certificates=args.root_certificates,
            private_key=args.private_key,
            certificate_chain=args.certificate_chain)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "maskrcnn"
    model_outputs = []
    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', [1, 3, 800, 1200], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data.reshape((1, 3, 800, 1200)))
    outputs.append(grpcclient.InferRequestedOutput('labels'))
    outputs.append(grpcclient.InferRequestedOutput('dets'))
    outputs.append(grpcclient.InferRequestedOutput('masks'))

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=args.client_timeout,
        headers={'test': '1'},
        compression_algorithm=args.grpc_compression_algorithm)

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    
    # Get the output arrays from the results
    dets_data = results.as_numpy('dets')
    labels_data = results.as_numpy('labels')
    masks_data = results.as_numpy('masks')
    
    print(' PASS: infer ' + model_name)

    # postprocess
    data_process.postprocess(dets_data[0], labels_data[0], masks_data[0], result_file)