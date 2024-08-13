import socket
import cv2
import numpy as np
import csv
import os
import shutil
import onnxruntime as ort
from PIL import Image
import argparse
import time
import random
#đống này dùng để detect biển báo
cuda = True
w = "best1.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
names = ['Straight', 'NoLeft', 'NoOpposite', 'NoRight', 'NoStraight', 'RoundRight', 'TurnLeft', 'TurnRight']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

count = 1
path = os.getcwd()
global sendBack_angle, sendBack_Speed, current_speed, current_angle,angle_change,speed_change
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
count = 0
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))
parser = argparse.ArgumentParser()
parser.add_argument('--map', type=int, default=3, help='select map')
args = parser.parse_args()
map = args.map
#tạo box
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def Control(angle, speed):
	global sendBack_angle, sendBack_Speed
	sendBack_angle = angle
	sendBack_Speed = speed

with open('./driving_log.csv','w',newline='') as f:
	writer = csv.writer(f)
	if __name__ == "__main__":
		try:
			while True:
				message_getState = bytes("0", "utf-8")
				s.sendall(message_getState)
				state_date = s.recv(100)

				try:
					current_speed, current_angle = state_date.decode("utf-8").split(' ')
				except Exception as er:
					print(er)
					pass

								
				message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
				s.sendall(message)
				data = s.recv(100000)

				try:
					image = cv2.imdecode(
						np.frombuffer(
							data,
							np.uint8
							), -1
						)
					# đống này lấy ảnh vô cái detect biến báo
					img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					imagee = img.copy()
					imagee, ratio, dwdh = letterbox(imagee, auto=False)
					imagee = imagee.transpose((2, 0, 1))
					imagee = np.expand_dims(imagee, 0)
					imagee = np.ascontiguousarray(imagee)
					im = imagee.astype(np.float32)
					im /= 255
					outname = [i.name for i in session.get_outputs()]

					inname = [i.name for i in session.get_inputs()]

					inp = {inname[0]:im}
					# ONNX inference
					outputs = session.run(outname, inp)[0]
					print(outputs,outputs.shape)
					ori_images = [img.copy()]

					for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
						imag = ori_images[int(batch_id)]
						box = np.array([x0,y0,x1,y1])
						box -= np.array(dwdh*2)
						box /= ratio
						box = box.round().astype(np.int32).tolist()
						cls_id = int(cls_id)
						score = round(float(score),3)
						name = names[cls_id]
						color = colors[name]
						name += ' '+str(score)
						print(cls_id, score)
						cv2.rectangle(imag,box[:2],box[2:],color,2)
						cv2.putText(imag,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

					ii = Image.fromarray(ori_images[0])

					open_cv_image = np.array(ii) 
					# Convert RGB to BGR 
					open_cv_image = open_cv_image[:, :, ::-1].copy() 
					cv2.imshow('1', open_cv_image)
					cv2.waitKey(1)	

					#oke đống dưới thì tính cái góc theo cái model
					speed = float(current_speed)
					angle = float(current_angle)
					#xử lý hình ảnh với ero và dilate
					imgg = cv2.blur(image,(5,5)) 
					hsv = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV)
					if map ==1:
						mask1 = cv2.inRange(hsv, (20, 0, 170), (110, 25,195))
						mask2 = cv2.inRange(hsv, (95, 80, 70), (115, 105,85))
						mask3 = cv2.inRange(hsv, (80, 0, 90), (115, 60,120))
						mask = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask2))
					else:
						mask1 = cv2.inRange(hsv, (95, 80, 65), (115, 105,82))
						mask2 = cv2.inRange(hsv, (95, 27, 115), (115, 70,145))
						mask3 = cv2.inRange(hsv, (95, 40, 85), (115, 80,120))
						mask4 = cv2.inRange(hsv, (95, 6, 150), (115, 50,180))
						mask123 = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask2))
						mask = cv2.bitwise_or(mask123, mask4)
					img_masked = cv2.bitwise_and(imgg,imgg, mask=mask)
					gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)	
					(thresh, bi_img) = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
					kernel = np.ones((5,5), np.uint8)
					img1 = cv2.morphologyEx(bi_img, cv2.MORPH_CLOSE, kernel)

					kernel = np.ones((7,7), np.uint8)
					img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

					kernel = np.ones((11,11), np.uint8)
					img3 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
					image = cv2.resize(src=img3, dsize=(320, 160))
					
					cv2.imshow("IMG", image)
					cv2.waitKey(1)	

				except Exception as er:
					print(er)
					pass

		finally:
			print('closing socket')
			s.close()
