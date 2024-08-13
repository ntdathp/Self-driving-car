import socket
import cv2
import numpy as np
import csv
import os
import shutil
import argparse
try:
	os.mkdir("IMG")
except:
	shutil.rmtree("IMG")	
	os.mkdir("IMG")
import random

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
					speed = float(current_speed)
					angle = float(current_angle)
					#xử lý hình ảnh với ero và dilate
					img = cv2.blur(image,(5,5)) 
					hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
					if map ==1:
						mask1 = cv2.inRange(hsv, (20, 0, 170), (110, 25,195))
						mask2 = cv2.inRange(hsv, (95, 80, 70), (115, 105,85))
						mask3 = cv2.inRange(hsv, (80, 0, 90), (115, 60,120))
						mask = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask2))
					else:
						mask1 = cv2.inRange(hsv, (95, 80, 65), (115, 105,82))
						mask2 = cv2.inRange(hsv, (95, 27, 115), (115, 70,145))
						mask3 = cv2.inRange(hsv, (95, 40, 80), (115, 80,120))
						mask4 = cv2.inRange(hsv, (95, 6, 150), (115, 50,180))
						mask123 = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask2))
						mask = cv2.bitwise_or(mask123, mask4)

					img_masked = cv2.bitwise_and(img,img, mask=mask)
					gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)	
					(thresh, bi_img) = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
					kernel = np.ones((5,5), np.uint8)
					img1 = cv2.morphologyEx(bi_img, cv2.MORPH_CLOSE, kernel)

					kernel = np.ones((7,7), np.uint8)
					img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

					kernel = np.ones((11,11), np.uint8)
					img3 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
					image = cv2.resize(src=img3, dsize=(320, 160))
					if (angle!=0 or random.random()<0.2) and abs(speed)>10:
						#chỉ lấy giá trị khi góc và vận tốc khác 0 để giảm mẫu
						cv2.imwrite("IMG/frame%d.jpg" % count, image) 
						link_image = os.path.join(path, "IMG", "frame%d.jpg" % count)
						writer.writerow([link_image, current_speed, current_angle])
						count += 1
						cv2.imshow("IMG", image)
						cv2.waitKey(1)	

				except Exception as er:
					print(er)
					pass

		finally:
			print('closing socket')
			s.close()
