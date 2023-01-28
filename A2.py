import cv2
import numpy as np
import mediapipe as mp

width = 1280
height = 720
fps = 30

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, fps)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


class Hands():
    import mediapipe as mp
    def __init__(self):
        self.hands = self.mp.solutions.hands.Hands(False,1,.5,.5)
        self.handDarw = mp.solutions.drawing_utils
    def HandData (self,frame,show_landmarks = True):
        landmarks = []
        label = []
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for i in results.multi_handedness:
                for j in i.classification:
                    label.append(j.label)
            for handlandmarks in results.multi_hand_landmarks:
                Landmarks = []
                if show_landmarks:
                    self.handDarw.draw_landmarks(frame, handlandmarks,mp.solutions.hands.HAND_CONNECTIONS)
                for landmark in handlandmarks.landmark:
                    Landmarks.append((int(width *landmark.x),int(height * landmark.y)))
                landmarks.append(Landmarks)
        return landmarks,label

landmarks = Hands()
name = 'anonymousAsquare'

cal_pos1 = (20+200,20)
cal_pos2 = (352+200,550)
cal_color = (237,237,237)
width_rec = 78
height_rec = 48
spacing = int(width_rec*0.06)
rec_pos1 = (int(cal_pos1[0]+ spacing),int(cal_pos1[1] + ((cal_pos2[1]-cal_pos1[1])*0.4)))
rec_pos2 = (int(rec_pos1[0]+width_rec),int(rec_pos1[1]+height_rec))
rec_color = (255,255,255)

class Calculator():
    def __init__(self):
        self.calc_input = ['','','']
        self.calc_output = ''
        self.input1 = True
        self.input3 = False
        self.output = False
        self.erase = False
        self.inverse_out = False
        self.inverse_output = ''
        self.square_out = False
        self.square_root_out = False
        self.square_output = ''
        self.square_root_output = ''

    def hand_dist(self,hand_data):
        if hand_data != []:
            x1 = hand_data[0][8][0]
            x2 = hand_data[0][12][0]

            y1 = hand_data[0][8][1]
            y2 = hand_data[0][12][1]

            distance = (((x2 - x1)**2) + ((y2-y1)**2))**(1/2)
            return distance

    def draw_calculator(self,frame,hand_data):
        calc_pos1 = (20+200,20)
        calc_pos2 = (352+200,550)
        cal_color = (237,237,237)
        width_rec = 78
        height_rec = 48
        spacing = int(width_rec*0.06)
        rec_pos1 = (int(cal_pos1[0]+ spacing),int(cal_pos1[1] + ((cal_pos2[1]-cal_pos1[1])*0.4)))
        rec_pos2 = (int(rec_pos1[0]+width_rec),int(rec_pos1[1]+height_rec))
        rec_color = (255,255,255)
        if hand_data != []:
            data = hand_data[0][12]
        else:
            data = []
        cv2.rectangle(frame, calc_pos1,calc_pos2, cal_color, -1)
        cols = 4
        rows = 6
        text_col = (0,0,0)
        rect_color = rec_color
        numbers = [0,1,2,3,4,5,6,7,8,9,'.']
        operators = ['+','-','/','x']
        characters = [['%','CE','C','X'],['1/x','x^2','sqrt','/'],['7','8','9','x'],['4','5','6','-'],['1','2','3','+'],['+/-','0','.','=']]
        for row in range(rows):
            for col in range(cols):
                text_col = (0,0,0)
                rect_color = rec_color
                pos1 = (rec_pos1[0]+(width_rec*col) + (spacing*col),rec_pos1[1]+(height_rec*row)+(spacing*row))
                pos2 = (rec_pos2[0]+(width_rec*col) + (spacing*col),rec_pos2[1]+(height_rec*row)+(spacing*row))
                if characters[row][col] == '=':
                    rect_color = (230,200,150)
                else:
                    rect_color = rec_color
                
                if data != []:
                    finger_distance = self.hand_dist(hand_data)
                    if finger_distance > 32:
                        if data[0] > pos1[0] and data[0] < pos2[0]:
                            if data[1] > pos1[1] and data[1] < pos2[1]:
                                rect_color = (155,155,155)
                                text_col = (255,255,255)

                    elif finger_distance <= 32:
                        if data[0] > pos1[0] and data[0] < pos2[0]:
                            if data[1] > pos1[1] and data[1] < pos2[1]:
                                rect_color = (0,0,0)
                                text_col = (255,255,255)
                                if self.input1:
                                    for i in numbers:
                                        if characters[row][col] == str(i):
                                            if len(self.calc_input[0]) == 0:
                                                self.calc_input[0] += (characters[row][col])
                                                self.erase = True
                                                # print(self.calc_input[0])
                                            else:
                                                if characters[row][col] != self.calc_input[0][len(self.calc_input[0])-1]:
                                                    if len(self.calc_input[0]) < 8:
                                                        self.calc_input[0] += (characters[row][col])
                                                        self.erase = True
                                                        # print(self.calc_input[0])
                                    
                                    if characters[row][col] == 'X':
                                        # print(self.erase)
                                        if len(self.calc_input[0]) != 0:
                                            if self.erase == True:
                                                self.calc_input[0] = self.calc_input[0].replace(self.calc_input[0][-1],"")
                                                self.erase = False
                                    
                                    if characters[row][col] == 'CE' or characters[row][col] == 'C':
                                        self.calc_input[0] = ""
                                
                                for i in operators:
                                    if characters[row][col] == i:
                                        self.calc_input[1] = i
                                        self.input1 = False
                                        self.input3 = True
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False
                                        if len(self.calc_input[2]) != 0:
                                            self.calc_input[0] = str(self.calculate(self.calc_input[0],self.calc_input[1],self.calc_input[2]))
                                            self.calc_input[2] = ''
                            
                                
                                if characters[row][col] == 'x^2':
                                    self.input3 = False
                                    self.input1 = False
                                    self.output = False
                                    self.square_out = True
                                    self.square_root_out = False
                                    self.inverse_out = False
                                    if len(self.calc_input[0]) == 0:
                                        self.calc_input[0] = '0'
                                    self.square_output = str(self.square(self.calc_input[0]))
                                
                                if characters[row][col] == 'sqrt':
                                    self.input3 = False
                                    self.input1 = False
                                    self.output = False
                                    self.square_out = False
                                    self.square_root_out = True
                                    self.inverse_out = False
                                    if len(self.calc_input[0]) == 0:
                                        self.calc_input[0] = '0'
                                    self.square_root_output = str(self.square_root(self.calc_input[0]))
                                
                                if characters[row][col] == '1/x':
                                    self.input3 = False
                                    self.input1 = False
                                    self.output = False
                                    self.square_out = False
                                    self.square_root_out = False
                                    self.inverse_out = True
                                    if len(self.calc_input[0]) == 0:
                                        self.calc_input[0] = '0'
                                    self.inverse_output = str(self.inverse(self.calc_input[0]))

                                if self.input3:
                                    for i in numbers:
                                        if characters[row][col] == str(i):
                                            if len(self.calc_input[2]) == 0:
                                                self.calc_input[2] += (characters[row][col])
                                                self.erase = True
                                                
                                            else:
                                                if characters[row][col] != self.calc_input[2][len(self.calc_input[2])-1]:
                                                    if len(self.calc_input[2]) < 8:
                                                        self.calc_input[2] += (characters[row][col])
                                                        self.erase = True
                                                        # print(self.calc_input[2])
                                    
                                    if characters[row][col] == 'X':
                                        # print(self.erase)
                                        if len(self.calc_input[2]) != 0:
                                            if self.erase == True:
                                                self.calc_input[2] = self.calc_input[2].replace(self.calc_input[2][-1],"")
                                                self.erase = False
                                    
                                    if characters[row][col] == 'CE':
                                        self.calc_input[2] = ""
                                    
                                    if  characters[row][col] == 'C':
                                        self.calc_input = ['','','']
                                        self.calc_output = ''
                                        self.input1 = True
                                        self.input3 = False
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False
                                
                                if characters[row][col] == '=':
                                    if len(self.calc_input[1]) != 0 :
                                        if len(self.calc_input[0]) == 0:
                                            self.calc_input[0] = '0'
                                        if len(self.calc_input[2]) == 0:
                                            self.calc_input[2] = '0'
                                        self.input3 = False
                                        self.output = True
                                        self.calc_output = str(self.calculate(self.calc_input[0],self.calc_input[1],self.calc_input[2]))
                                
                                if self.output:
                                    if characters[row][col] == 'CE' or characters[row][col] == 'C':
                                        self.calc_input = ['','','']
                                        self.calc_output = ''
                                        self.input1 = True
                                        self.input3 = False
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False

                                if self.square_out:
                                    if characters[row][col] == 'CE' or characters[row][col] == 'C':
                                        self.calc_input = ['','','']
                                        self.calc_output = ''
                                        self.input1 = True
                                        self.input3 = False
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False
                                
                                if self.square_root_out:
                                    if characters[row][col] == 'CE' or characters[row][col] == 'C':
                                        self.calc_input = ['','','']
                                        self.calc_output = ''
                                        self.input1 = True
                                        self.input3 = False
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False
                                
                                if self.inverse_out:
                                    if characters[row][col] == 'CE' or characters[row][col] == 'C':
                                        self.calc_input = ['','','']
                                        self.calc_output = ''
                                        self.input1 = True
                                        self.input3 = False
                                        self.output = False
                                        self.square_out = False
                                        self.square_root_out = False
                                        self.inverse_out = False
                                
            
                cv2.rectangle(frame,pos1,pos2, rect_color, -1)
                if len(characters[row][col]) == 1:
                    char_pos = (pos1[0]+int(width_rec/2)-int((width_rec/2)/2/2),pos1[1]+height_rec-int((height_rec/2)/2))
                elif(len(characters[row][col]) == 2):
                    char_pos = (pos1[0]+int(width_rec/2)-int((width_rec/2)/2),pos1[1]+height_rec-int((height_rec/2)/2))
                elif(characters[row][col] == '+/-'):
                    char_pos = (pos1[0]+int(width_rec/2)-int((width_rec/2))+int((width_rec/5)),pos1[1]+height_rec-int((height_rec/2)/2))
                else:
                    char_pos = (pos1[0]+int(width_rec/2)-int((width_rec/2))+int((width_rec/5)),pos1[1]+height_rec-int((height_rec/2)/2))
                cv2.putText(frame,characters[row][col],char_pos,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,text_col,1)
                cv2.putText(frame,'calculator',(calc_pos1[0]+10,cal_pos1[1]+20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1)
                cv2.putText(frame,'_',(calc_pos1[0]+10,cal_pos1[1]+35),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                cv2.putText(frame,'_',(calc_pos1[0]+10,cal_pos1[1]+40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                cv2.putText(frame,'_',(calc_pos1[0]+10,cal_pos1[1]+45),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                cv2.putText(frame,'Standard',(calc_pos1[0]+45,cal_pos1[1]+50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                #cv2.putText(frame,calc_input[0],((calc_pos2[0]-10)-10*len(calc_input[0]),cal_pos1[1]+200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
                if self.input1:
                    for i in range(0,len(self.calc_input[0]),1):
                        cv2.putText(frame,self.calc_input[0][(len(self.calc_input[0])-1)-i],(calc_pos2[0]-40-i*40,cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
                if self.input3:
                    for i in range(0,len(self.calc_input[2]),1):
                        cv2.putText(frame,self.calc_input[2][(len(self.calc_input[2])-1)-i],(calc_pos2[0]-40-i*40,cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
                    if (len(self.calc_input[0]) == 0):
                        text = '0'
                    else:
                        text = self.calc_input[0]+self.calc_input[1]
                    cv2.putText(frame,text,(calc_pos2[0]-((len(self.calc_input[0])+1)*15),cal_pos1[1]+120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)

                if self.output:
                   text1 = self.calc_input[0]+self.calc_input[1]+self.calc_input[2]
                   cv2.putText(frame,text1,(calc_pos2[0]-((len(self.calc_input[0])+len(self.calc_input[2])+1)*15),cal_pos1[1]+120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                   text2 = self.calc_output
                   cv2.putText(frame,text2,(calc_pos2[0]-(len(self.calc_output)*40),cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
                
                if self.square_out:
                    text1 = 'Sq({})'.format(self.calc_input[0])
                    cv2.putText(frame,text1,(calc_pos2[0]-((len(self.calc_input[0])+2+1)*15),cal_pos1[1]+120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                    text2 = self.square_output
                    cv2.putText(frame,text2,(calc_pos2[0]-(len(self.square_output)*40),cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
                
                if self.square_root_out:
                    text1 = 'Sqrt({})'.format(self.calc_input[0])
                    cv2.putText(frame,text1,(calc_pos2[0]-((len(self.calc_input[0])+4+1)*15),cal_pos1[1]+120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                    text2 = self.square_root_output
                    cv2.putText(frame,text2,(calc_pos2[0]-(len(self.square_root_output)*37),cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
                
                if self.inverse_out:
                    text1 = '1/({})'.format(self.calc_input[0])
                    cv2.putText(frame,text1,(calc_pos2[0]-((len(self.calc_input[0])+2+1)*15),cal_pos1[1]+120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
                    text2 = self.inverse_output
                    cv2.putText(frame,text2,(calc_pos2[0]-(len(self.inverse_output)*37),cal_pos1[1]+200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)


    def calculate(self,first,operator,second):
        decimal_a = False
        decimal_b = False
        for i in first:
            if i == '.':
                if decimal_a != True:
                    decimal_a = True 
        
        for i in second:
            if i == '.':
                if decimal_b != True:
                    decimal_b = True
        
        if decimal_a:
            a = float(first)

        elif decimal_a == False:
            a = int(first)

        if decimal_b:
            b = float(second)

        elif decimal_b == False:
            b = int(second)

        if operator == '+':
            answer = a + b  

        elif operator == '-':
            answer = a - b

        elif operator == '/':
            answer = a / b   

        elif operator == 'x':
            answer = a * b  
        
        return round(answer,3)

    def square_root(self,num):
        decimal_num = False
        for i in num:
            if i == '.':
                if decimal_num != True:
                    decimal_num = True 
        
        if decimal_num:
            num = float(num)

        elif decimal_num == False:
            num = int(num)
        ans = (num)**(1/2)
        return round(ans,5)
    
    def square(self,num):
        decimal_num = False
        for i in num:
            if i == '.':
                if decimal_num != True:
                    decimal_num = True 
        
        if decimal_num:
            num = float(num)

        elif decimal_num == False:
            num = int(num)
        ans = (num)**2
        return round(ans,5)

    def inverse(self,num):
        decimal_num = False
        for i in num:
            if i == '.':
                if decimal_num != True:
                    decimal_num = True 
        
        if decimal_num:
            num = float(num)

        elif decimal_num == False:
            num = int(num)
        ans = 1/(num)
        return round(ans,5)

class facelm():
    import mediapipe as mp
    def __init__(self):
        self.face = self.mp.solutions.face_mesh.FaceMesh(False,2,.5,.5)
        self.draw_face = self.mp.solutions.drawing_utils
        self.drawSpecC = self.draw_face.DrawingSpec(thickness = 0,circle_radius =0, color = (255,255,255))
        self.drawSpecL = self.draw_face.DrawingSpec(thickness = 3,circle_radius =2, color = (0,255,0))

    def landmarks(self,frame):
        landmarks = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face.process(frameRGB)
        if (results.multi_face_landmarks) != None:
            for lms in results.multi_face_landmarks:
                self.draw_face.draw_landmarks(frame,lms,self.mp.solutions.face_mesh.FACE_CONNECTIONS, self.drawSpecC,self.drawSpecL)
                landmark = []
                for lm in lms.landmark:
                    landmark.append((int(width*lm.x),int(height*lm.x)))
                landmarks.append(landmark)
        return landmarks

class Face_detection():
    import mediapipe as mp
    def __init__(self):
        self.face = self.mp.solutions.face_detection.FaceDetection()
        self.draw = self.mp.solutions.drawing_utils
    def faceD (self,frame):
        Faces= []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face.process(frameRGB)
        if (results.detections != None):
            face = []
            for faces in results.detections: 
                i = (faces.location_data.relative_bounding_box)
                face.append((int(i.xmin*width),int(i.ymin*height),int(i.width*width),int(i.height*height)))
            Faces.append(face)
        return Faces

calculator = Calculator()
facelandm = facelm()
face = Face_detection()

while True:
    ignore, frame = cam.read()

    data,label = landmarks.HandData(frame)
    calculator.draw_calculator(frame, data)
    # facelandm.landmarks(frame)
    faces = face.faceD(frame)
    # for i in faces:
    #     for j in i:
    #         x,y,w,h = j
    #         print(i)
    #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    facelandm.landmarks(frame)
    
    cv2.imshow(name, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release() 