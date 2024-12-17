{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0a67efca-1455-4bd5-916e-ccd75ba13771",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "import yaml\n",
    "from yaml.loader import SafeLoader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1e5da22e-eb1f-490d-be67-b3f938edc81a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['person', 'car', 'chair', 'bottle', 'pottedplant', 'bird', 'dog', 'sofa', 'bicycle', 'horse', 'boat', 'motorbike', 'cat', 'tvmonitor', 'cow', 'sheep', 'aeroplane', 'train', 'diningtable', 'bus']\n"
     ]
    }
   ],
   "source": [
    "# Load YAML\n",
    "with open('data.yaml', mode='r') as f:\n",
    "    data_yaml = yaml.load(f, Loader=SafeLoader)\n",
    "\n",
    "labels = data_yaml['names']\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "da801ec1-0c01-42ac-9067-68d9155f931a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load YOLO Model\n",
    "yolo = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')\n",
    "yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)\n",
    "yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a340d460-2a20-4866-9524-970a2c8f0f73",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the image\n",
    "img = cv2.imread('./street_image.jpg')\n",
    "image = img.copy()\n",
    "row, col, d = image.shape\n",
    "\n",
    "# get the YOLO prediction from the image\n",
    "# step-1 convert image into square image (array)\n",
    "max_rc = max(row, col)\n",
    "input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)\n",
    "input_image[0: row, 0: col] = image\n",
    "# step-2 get prediction from square array\n",
    "INPUT_WH_YOLO = 640\n",
    "blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=True)\n",
    "yolo.setInput(blob)\n",
    "preds = yolo.forward() # detection or prediction from YOLO"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a08896f8-b60c-4a9b-9d25-e0ba233c1d4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1, 25200, 25)\n"
     ]
    }
   ],
   "source": [
    "print(preds.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "80c7f3de-179c-4304-b56c-dd8dedebc87a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Non Maximum Suppression\n",
    "# step-1: filter detection based on confidence (0.4) and probability score (0.25)\n",
    "detections = preds[0]\n",
    "boxes = []\n",
    "confidences = []\n",
    "classes = []\n",
    "\n",
    "# width and height of the image (input_image)\n",
    "image_w, image_h = input_image.shape[:2]\n",
    "x_factor = image_w / INPUT_WH_YOLO\n",
    "y_factor = image_h / INPUT_WH_YOLO\n",
    "\n",
    "for i in range(len(detections)):\n",
    "    row = detections[i]\n",
    "    confidence = row[4] # confidence of detection an object\n",
    "    if confidence > 0.4:\n",
    "        class_score = row[5:].max() # maximum probability from 20 objects\n",
    "        class_id = row[5:].argmax() # get the index position at which max probability occur\n",
    "\n",
    "        if class_score > 0.25:\n",
    "            cx, cy, w, h = row[0:4]\n",
    "            # construct bounding box from four values\n",
    "            # left, top, width and height\n",
    "            left = int((cx - 0.5 * w) * x_factor)\n",
    "            top = int((cy - 0.5 * h) * y_factor)\n",
    "            width = int(w * x_factor)\n",
    "            height = int(h * y_factor)\n",
    "\n",
    "            box = np.array([left, top, width, height])\n",
    "\n",
    "            # append values into the\n",
    "            confidences.append(confidence)\n",
    "            boxes.append(box)\n",
    "            classes.append(class_id)\n",
    "\n",
    "# clean\n",
    "boxes_np = np.array(boxes).tolist()\n",
    "confidences_np = np.array(confidences).tolist()\n",
    "\n",
    "# NMS\n",
    "index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "33915c8b-7b97-41ce-a84c-19f1ac49d27c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d1060b63-4f6a-4200-936b-196929e55e0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Draw the Bounding\n",
    "for ind in index:\n",
    "    # extract bounding box\n",
    "    x, y, w, h = boxes_np[ind]\n",
    "    bb_conf = int(confidences_np[ind] * 100)\n",
    "    classes_id = classes[ind]\n",
    "    class_name = labels[classes_id]\n",
    "\n",
    "    text = f'{class_name}: {bb_conf}%'\n",
    "    \n",
    "    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)\n",
    "\n",
    "    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6513f9e9-1b38-4981-b868-db283b426445",
   "metadata": {},
   "outputs": [],
   "source": [
    "cv2.imshow('original', img)\n",
    "cv2.imshow('yolo_prediction', image)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03d4d6c8-0f99-40ff-932b-9ad56a72f6de",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
