# OpenCV 실습 - 움직이는 물체 감지로 침입탐지 밑 저장

# 200601

# https://potatoggg.tistory.com/192

# http://study.marearts.com/2017/02/cvlecture-example-code-video-subtraction.html

# http://cvlecture.marearts.com/2017/02/opencv-4-5.html


import cv2
import datetime


def main():
	cap = cv2.VideoCapture(1)  # 0: default camera
	#cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기

	min_area = 8000  # 움직이는 contour중 8000 이상인 물체만 인식, 작은건 무시

	old_gray = 0
	frame_cnt = 0
	while cap.isOpened():
	    # 카메라 프레임 읽기
	    success, frame = cap.read()  # frame : (480, 640, 3)
	    if success:
	        frame_cnt += 1
	        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	        gray = cv2.GaussianBlur(gray, (21, 21), 0)
	        fps = cap.get(cv2.CAP_PROP_FPS)
	        if frame_cnt % 30 == 0:
	            print('fps : %d'%(fps))
	        
	        if type(old_gray)==int:
	            old_gray = gray
	        
	        
	        substraction_frame = cv2.absdiff(gray, old_gray)
	        _, abs_diff_thresolding =  cv2.threshold(substraction_frame, 20, 255, cv2.THRESH_BINARY)
	        dilated_abs_diff_thresolding = cv2.dilate(abs_diff_thresolding, None, iterations=2)  # 팽창
	        cnts, _ = cv2.findContours(dilated_abs_diff_thresolding.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	        
	        # 사진에서 contour 갯수만큼 loop
	        count2 = 0
	        if (frame_cnt % 3 == 0):
	            for c in cnts:
	                count2 += 1
	                if cv2.contourArea(c) < min_area:
	                    continue
	                # contour를 위한 bbox 그리기
	                (x, y, w, h) = cv2.boundingRect(c)
	                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	                img_trim = frame[y : y+h, x : x+w]  # 이미지 자르기
	                cv2.imwrite('save/%d.jpg'%(frame_cnt), img_trim)
	        
	        cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'), (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	        
	        # 프레임 출력
	        cv2.imshow('Camera Window', frame)
	        cv2.imshow('Difference_dilate', dilated_abs_diff_thresolding)
	        
	        old_gray = gray

	        # ESC를 누르면 종료
	        if (cv2.waitKey(1) & 0xFF == 27):
	            cap.release()
	            cv2.destroyAllWindows()
	            break

	cap.release()
	cv2.destroyAllWindows()



main()