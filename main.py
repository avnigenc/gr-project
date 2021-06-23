import numpy as np
import cv2
from imutils.video import FPS
import json
import redis
from pymongo import MongoClient

APP_CLIENT_VERSION = 1

redis_client = redis.Redis()
redis_client.flushall()

mongo_client = MongoClient('mongodb://localhost:27017/')
mongoDatabase = mongo_client.graduationProject
sessionCollection = mongoDatabase.sessions

INPUT_FILE = 'input/sample.mp4'
OUTPUT_FILE = 'output/output.avi'
LABELS_FILE = 'data/coco.names'
CONFIG_FILE = 'cfg/yolov3.cfg'
WEIGHTS_FILE = 'data/yolov3.weights'
CONFIDENCE_THRESHOLD = 0.3

H = None
W = None

fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30, (800, 600), True)

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
vs = cv2.VideoCapture(INPUT_FILE)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
count = 0
is_running = True
first_image = None
overlay = None
final_image = None
while True:
    count += 1
    print("[INFO] Frame number", count)
    try:
        (grabbed, image) = vs.read()
        if first_image is None:
            first_image = cv2.resize(image.copy(), (800, 600))
            overlay = cv2.resize(image.copy(), (800, 600))
    except:
        break

    if image is None:
        break
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    if W is None or H is None:
        (H, W) = image.shape[:2]
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            if LABELS[classIDs[i]] == 'person':
                redis_object = json.dumps({
                    'APP_CLIENT_VERSION': APP_CLIENT_VERSION,
                    'x': (x, y),
                    'y': (x + w, y + h)
                })
                redis_client.set(count, redis_object)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Avni Genc - Graduation Project (DEMO)", cv2.resize(image, (800, 600)))
    writer.write(cv2.resize(image, (800, 600)))
    fps.update()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

alpha = 0.02
redis_object_keys = redis_client.keys()
for i in redis_object_keys:
    redis_object = redis_client.get(i)
    json_redis_object = json.loads(redis_object)
    cv2.rectangle(overlay, json_redis_object['x'], json_redis_object['y'], (244, 213, 252, 0.05), -1)
    sessionCollection.insert_one(json_redis_object)
    first_image = cv2.addWeighted(overlay, alpha, first_image, 1-alpha, 0)

if first_image is not None:
    cv2.imwrite('output/FinalResult.jpg', cv2.resize(first_image, (800, 600)))

cv2.imshow('Avni Genc - Graduation Project (DEMO)', first_image)
cv2.waitKey()

print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
writer.release()
vs.release()
