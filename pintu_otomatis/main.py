
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:05:11 2020

@author: FAHMI-PC
"""

#import library

import os
import sys
import cv2
import csv

import smtplib
import mimetypes
import datetime
import numpy as np
import pandas as pd
import time
from do_something import *
import multiprocessing as jalan
from threading import Thread

from skimage import io
from email import encoders
from PIL import Image,ImageTk
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.message import EmailMessage
from tkinter.filedialog import askopenfilename
from email.mime.multipart import MIMEMultipart

import dlib                           
import shutil

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = mainScreen (root)
    root.mainloop()

w = None
def create_mainScreen(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = mainScreen (w)
    AMS_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_mainScreen():
    global w
    w.destroy()
    w = None

class mainScreen:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font9 = "-family {SF Pro Display} -size 14 -weight bold -slant"  \
            " roman -underline 0 -overstrike 0"
        font10 = "-family {SF Pro Display} -size 14 -weight bold "  \
            "-slant roman -underline 0 -overstrike 0"
        font11 = "-family {SF Pro Display} -size 14 -weight bold "  \
            "-slant roman -underline 0 -overstrike 0"
        font12 = "-family {SF Pro Display} -size 12 -weight bold "  \
            "-slant roman -underline 0 -overstrike 0"

        def takeImage():
        	detector_face = dlib.get_frontal_face_detector()
        	cap_face = cv2.VideoCapture(0)
        	cnt_ss_face = 0
        	current_face_dir = ""
        	path_photos_from_camera = "data/create/"

        	def ambyar_work_mkdir():
        		if os.path.isdir(path_photos_from_camera):
        			pass
        		else:
        			os.mkdir(path_photos_from_camera)
        	ambyar_work_mkdir()


        	def ambyar_work_del_old_face_folders():
        		folders_rd = os.listdir(path_photos_from_camera)
        		for i in range(len(folders_rd)):
        			shutil.rmtree(path_photos_from_camera+folders_rd[i])
        		if os.path.isfile("data/features_all.csv"):
        			os.remove("data/features_all.csv")

        	if os.listdir("data/create/"):
        		person_list = os.listdir("data/create/")
        		person_num_list = []
        		for person in person_list:
        			person_num_list.append(int(person.split('_')[-1]))
        		person_cnt = max(person_num_list)

        	else:
        		person_cnt = 0
        	save_flag = 1
        	press_n_flag = 0

        	while cap_face.isOpened():
        		flag, img_rd = cap_face.read()
        		kk_ambyar = cv2.waitKey(1)
        		faces = detector_face(img_rd, 0)
        		font = cv2.FONT_ITALIC

        		if kk_ambyar == ord('n'):
        			person_cnt += 1
        			current_face_dir = path_photos_from_camera + "person_" + str(person_cnt)
        			os.makedirs(current_face_dir)
        			print('\n')
        			print(" Buat Folder: ", current_face_dir)
        			cnt_ss = 0
       				press_n_flag = 1

        		if len(faces) != 0:
        			for k, d in enumerate(faces):
        				pos_start = tuple([d.left(), d.top()])
        				pos_end = tuple([d.right(), d.bottom()])
        				height = (d.bottom() - d.top())
        				width = (d.right() - d.left())
        				hh = int(height/2)
        				ww = int(width/2)
        				color_rectangle = (255, 255, 255)

        				if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
        					cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        					color_rectangle = (0, 0, 255)
        					save_flag = 0
        					if kk_ambyar == ord('s'):
        						print(" Tolong sesuaikan di posisi")
        				else:
        					color_rectangle = (255, 255, 255)
        					save_flag = 1

        				cv2.rectangle(img_rd,tuple([d.left() - ww, d.top() - hh]),
        					tuple([d.right() + ww, d.bottom() + hh]),color_rectangle, 2)
        				img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

        				if save_flag:
        					if kk_ambyar == ord('s'):
        						if press_n_flag:
        							cnt_ss += 1
        							for ii in range(height*2):
        								for jj in range(width*2):
        									img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
        							cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", img_blank)
        							print(" Save into：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
        						else:
        							print(" Tolong Jika 'N' Sebelum 'S'")

        		cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        		cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        		cv2.putText(img_rd, "N: Create face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        		cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        		cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        		if kk_ambyar == ord('q'):
        			break

        		cv2.imshow("camera", img_rd)

        	cap_face.release()
        	cv2.destroyAllWindows()




        def trainImage():
            path_images_from_camera = "data/create/"
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
            face_rec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

            def return_128d_features(path_img):
                img_rd = io.imread(path_img)
                faces = detector(img_rd, 1)


                print("%-40s %-20s" % ("image with faces detected:", path_img), '\n')

    

                if len(faces) != 0:
                    shape = predictor(img_rd, faces[0])
                    face_descriptor = face_rec.compute_face_descriptor(img_rd, shape)
        
                else:
                    face_descriptor = 0
                    print("no face")

                return face_descriptor


            def return_features_mean_personX(path_faces_personX):
                features_list_personX = []
                photos_list = os.listdir(path_faces_personX)
    
                if photos_list:
                    for i in range(len(photos_list)):
            
                        print("%-40s %-20s" % ("image to read:", path_faces_personX + "/" + photos_list[i]))
                        features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
           
            
                        if features_128d == 0:
                            i += 1
                        else:
                            features_list_personX.append(features_128d)
                
                else:
                    print(" Warning: Tidak ada images di " + path_faces_personX + '/', '\n')

  
                if features_list_personX:
                    features_mean_personX = np.array(features_list_personX).mean(axis=0)
        
                else:
                    features_mean_personX = '0'

                return features_mean_personX


            person_list = os.listdir("data/create/")

            person_num_list = []

            for person in person_list:
    
                person_num_list.append(int(person.split('_')[-1]))
            person_cnt = max(person_num_list)

            with open("data/features_all.csv", "w", newline="") as csvfile:
    
                writer = csv.writer(csvfile)
    
                for person in range(person_cnt):
                    # Get the mean/average features of face/personX, it will be a list with a length of 128D
        
                    print(path_images_from_camera + "person_"+str(person+1))
        
                    features_mean_personX = return_features_mean_personX(path_images_from_camera + "person_"+str(person+1))
        
                    writer.writerow(features_mean_personX)
        
                    print("People of features:", list(features_mean_personX))
        
                    print('\n')
        
                print("Save all the features of faces registered into: data/features_all.csv")
    
    
    

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            IDS = []
            for imagePath in imagePaths:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(imageNp)
                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y + h, x:x + w])
                    IDS.append(Id)
            return faceSamples, IDS

        def autoAttendance():
            facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


            # fungsi ini sebagai between two 128D features
            def return_euclidean_distance(feature_1, feature_2):
                feature_1 = np.array(feature_1)
                feature_2 = np.array(feature_2)
                dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
                return dist

            # fungsi start pada thread
            def start(self):
                Thread(target=self.update, args=()).start()
                return self


            # melakukan cek csv
            if os.path.exists("data/features_all.csv"):
                path_features_known_csv = "data/features_all.csv"
                csv_rd = pd.read_csv(path_features_known_csv, header=None)

                # setelah itu 
                # array akan di save
                features_known_arr = []

                # melakukan print known faces
    
                for i in range(csv_rd.shape[0]):
                    features_someone_arr = []
                    for j in range(0, len(csv_rd.iloc[i])):
                        features_someone_arr.append(csv_rd.iloc[i][j])
                    features_known_arr.append(features_someone_arr)
                print("Faces in Database：", len(features_known_arr))

                # Dlib detejsu
                # detector dan predictor yang digunakan
                detector = dlib.get_frontal_face_detector()
    
                predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

                # munculkan webcam
                cap = cv2.VideoCapture(0)

                # jika webcam terbuka 
    
                while cap.isOpened():

                    #sebagai flag sebelumnya untuk read wajah
                    flag, img_rd = cap.read()
        
                    faces = detector(img_rd, 0)

                    # font untuk di tulis 
                    font = cv2.FONT_ITALIC

                    # list to save the posisi dan nama
                    pos_namelist = []
                    name_namelist = []

                    kk = cv2.waitKey(1)

                    # untuk menunggu 
                    # tekan 'q' untuk exit
                    if kk == ord('q'):
                        break
                    else:
                        # jika wajah terdeteksi
                        if len(faces) != 0:
                            # features_cap_arr
                            # capture dan save into features_cap_arr
                
                            features_cap_arr = []
                            for i in range(len(faces)):
                                shape = predictor(img_rd, faces[i])
                                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))


                            # convert to the database csv
                            for k in range(len(faces)):
                                print("##### camera person", k+1, "#####")
                                # 
                                # jika ada yang unknown
                                # Set the default names of faces with "unknown"
                                name_namelist.append("unknown")

                                # posisi di capture
                                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                                # face sudah di database
                    
                                e_distance_list = []
                    
                                for i in range(len(features_known_arr)):
                
                                    if str(features_known_arr[i][0]) != '0.0':
                                        print("with person", str(i + 1), "the e distance: ", end='')
                                        e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                                        print(e_distance_tmp)
                                        e_distance_list.append(e_distance_tmp)
                        
                                    else:
                          
                                        e_distance_list.append(999999999)
                            
                                # temukan minimal 1 person
                                similar_person_num = e_distance_list.index(min(e_distance_list))
                                print("Minimum e distance with person", int(similar_person_num)+1)

                                if min(e_distance_list) < 0.4:
                       
                                    # person1, 2, 3 .....
                        
                                    name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                                    print("May be person "+str(int(similar_person_num)+1))
                                else:
                                    print("Unknown person")

                                # ini sudah membaca person
                    
                 
                                for kk, d in enumerate(faces):
                        
                                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                                print('\n')


                            # menulis nama under rectangle
                            for i in range(len(faces)):
                                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                    print("Faces in camera now:", name_namelist, "\n")

                    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        
                    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
                    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow("camera", img_rd)

                cap.release()
    
                cv2.destroyAllWindows()


            #fungsi pool dari multiprocessing
    
            if __name__ == "__main__":
                pool = jalan.Pool(jalan.cpu_count()- 1)
                cap.release()
                cv2.destroyAllWindows()


            else:
                print('##### Warning #####', '\n')
                print("'features_all.py' not found!")
                print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')
                print('##### Warning #####')



        def manualAttendance():
            facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


            # fungsi ini sebagai between two 128D features
            def return_euclidean_distance(feature_1, feature_2):
                feature_1 = np.array(feature_1)
                feature_2 = np.array(feature_2)
                dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
                return dist

            # fungsi start pada thread
            def start(self):
                Thread(target=self.update, args=()).start()
                return self


            # melakukan cek csv
            if os.path.exists("data/features_all.csv"):
                path_features_known_csv = "data/features_all.csv"
                csv_rd = pd.read_csv(path_features_known_csv, header=None)

                # setelah itu 
                # array akan di save
                features_known_arr = []

                # melakukan print known faces
    
                for i in range(csv_rd.shape[0]):
                    features_someone_arr = []
                    for j in range(0, len(csv_rd.iloc[i])):
                        features_someone_arr.append(csv_rd.iloc[i][j])
                    features_known_arr.append(features_someone_arr)
                print("Faces in Database：", len(features_known_arr))

                # Dlib detejsu
                # detector dan predictor yang digunakan
                detector = dlib.get_frontal_face_detector()
    
                predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

                # munculkan webcam
                cap = cv2.VideoCapture(0)

                # jika webcam terbuka 
    
                while cap.isOpened():

                    #sebagai flag sebelumnya untuk read wajah
                    flag, img_rd = cap.read()
        
                    faces = detector(img_rd, 0)

                    # font untuk di tulis 
                    font = cv2.FONT_ITALIC

                    # list to save the posisi dan nama
                    pos_namelist = []
                    name_namelist = []

                    kk = cv2.waitKey(1)

                    # untuk menunggu 
                    # tekan 'q' untuk exit
                    if kk == ord('q'):
                        break
                    else:
                        # jika wajah terdeteksi
                        if len(faces) != 0:
                            # features_cap_arr
                            # capture dan save into features_cap_arr
                
                            features_cap_arr = []
                            for i in range(len(faces)):
                                shape = predictor(img_rd, faces[i])
                                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))


                            # convert to the database csv
                            for k in range(len(faces)):
                                print("##### camera person", k+1, "#####")
                                # 
                                # jika ada yang unknown
                                # Set the default names of faces with "unknown"
                                name_namelist.append("unknown")

                                # posisi di capture
                                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                                # face sudah di database
                    
                                e_distance_list = []
                    
                                for i in range(len(features_known_arr)):
                
                                    if str(features_known_arr[i][0]) != '0.0':
                                        print("with person", str(i + 1), "the e distance: ", end='')
                                        e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                                        print(e_distance_tmp)
                                        e_distance_list.append(e_distance_tmp)
                        
                                    else:
                          
                                        e_distance_list.append(999999999)
                            
                                # temukan minimal 1 person
                                similar_person_num = e_distance_list.index(min(e_distance_list))
                                print("Minimum e distance with person", int(similar_person_num)+1)

                                if min(e_distance_list) < 0.4:
                       
                                    # person1, 2, 3 .....
                        
                                    name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                                    print("May be person "+str(int(similar_person_num)+1))
                                else:
                                    print("Unknown person")

                                # ini sudah membaca person
                    
                 
                                for kk, d in enumerate(faces):
                        
                                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                                print('\n')


                            # menulis nama under rectangle
                            for i in range(len(faces)):
                                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                    print("Faces in camera now:", name_namelist, "\n")

                    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        
                    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
                    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow("camera", img_rd)

                cap.release()
    
                cv2.destroyAllWindows()


            #fungsi pool dari multiprocessing
    
            if __name__ == "__main__":
                pool = jalan.Pool(jalan.cpu_count()- 1)
                cap.release()
                cv2.destroyAllWindows()


            else:
                print('##### Warning #####', '\n')
                print("'features_all.py' not found!")
                print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')
                print('##### Warning #####')

        def adminPanel():
            adminScreen = tk.Tk()
            adminScreen.geometry("730x389+225+149")
            adminScreen.resizable(1, 1)
            adminScreen.title("Admin Panel")
            adminScreen.iconbitmap("mainIcon.ico")
            adminScreen.configure(background="#1B1B1B")
            adminScreen.focus_force()

            def clearUsername():
                self.adminUsernameEntry.delete(first=0, last=30)

            def clearPassword():
                self.adminPasswordEntry.delete(first=0, last=30)

            def administratorLogin():
                UserName = self.adminUsernameEntry.get()
                Password = self.adminPasswordEntry.get()
                if UserName == os.environ.get("panelUsername"):
                    if Password == os.environ.get("panelPassword"):
                        self.loginMessage.configure(background="#008000")
                        self.loginMessage.configure(foreground="#FFFFFF")
                        self.loginMessage.configure(text='''Login Success!''')
                        studentDetails = tk.Tk()
                        studentDetails.title("Student Details")
                        studentDetails.iconbitmap("mainIcon.ico")
                        studentDetails.configure(background="#1B1B1B")
                        studentDetails.focus_force()
                        location = 'D:/FOLDER KHUSUS NGAMPUS/SEMESTER 6/SISTEM TERSEBAR/UTS/UTS/FIX/Multiprocessing/Smart Absensi/Smart Absensi/StudentDetails.csv'
                        with open (location, newline="") as file:
                            reader = csv.reader(file)
                            r = 0
                            for col in reader:
                                c = 0
                                for row in col:
                                    self.studentLabel = tk.Label(studentDetails)
                                    self.studentLabel.configure(background="#008000")
                                    self.studentLabel.configure(foreground="#000000")
                                    self.studentLabel.configure(font="-family {SF Pro Display} -size 18 -weight bold")
                                    self.studentLabel.configure(width=6, height=1)
                                    self.studentLabel.configure(text=row)
                                    self.studentLabel.grid(row = r, column = c)
                                    c += 1
                                r += 1
                        adminScreen.iconify()
                        studentDetails.mainloop()
                    elif Password == "":
                        self.loginMessage.configure(background="#800000")
                        self.loginMessage.configure(foreground="#FFFFFF")
                        self.loginMessage.configure(text='''Please enter password!''')
                    else:
                        self.loginMessage.configure(background="#800000")
                        self.loginMessage.configure(foreground="#FFFFFF")
                        self.loginMessage.configure(text='''Incorrect Password!''')
                        clearPassword()
                elif UserName == "":
                    self.loginMessage.configure(background="#800000")
                    self.loginMessage.configure(foreground="#FFFFFF")
                    self.loginMessage.configure(text='''Please enter username!''')
                else:
                    self.loginMessage.configure(background="#800000")
                    self.loginMessage.configure(foreground="#FFFFFF")
                    self.loginMessage.configure(text='''Incorrect Username!''')
                    clearUsername()

            self.topMessage = tk.Message(adminScreen)
            self.topMessage.place(relx=0.0, rely=0.051, relheight=0.175, relwidth=1.041)
            self.topMessage.configure(background="#2E2E2E")
            self.topMessage.configure(font="-family {SF Pro Display} -size 36 -weight bold")
            self.topMessage.configure(foreground="#FFFFFF")
            self.topMessage.configure(highlightbackground="#d9d9d9")
            self.topMessage.configure(highlightcolor="black")
            self.topMessage.configure(text='''Admin Panel''')
            self.topMessage.configure(width=760)

            self.adminUsername = tk.Label(adminScreen)
            self.adminUsername.place(relx=0.096, rely=0.36, height=29, width=155)
            self.adminUsername.configure(background="#1B1B1B")
            self.adminUsername.configure(disabledforeground="#a3a3a3")
            self.adminUsername.configure(font=font10)
            self.adminUsername.configure(foreground="#FFFFFF")
            self.adminUsername.configure(text='''Enter Username:''')

            self.adminPassword = tk.Label(adminScreen)
            self.adminPassword.place(relx=0.096, rely=0.54, height=29, width=152)
            self.adminPassword.configure(background="#1B1B1B")
            self.adminPassword.configure(disabledforeground="#a3a3a3")
            self.adminPassword.configure(font=font10)
            self.adminPassword.configure(foreground="#FFFFFF")
            self.adminPassword.configure(text='''Enter Password:''')

            self.adminUsernameEntry = tk.Entry(adminScreen)
            self.adminUsernameEntry.place(relx=0.384, rely=0.36, height=27, relwidth=0.362)
            self.adminUsernameEntry.configure(background="#D9D9D9")
            self.adminUsernameEntry.configure(disabledforeground="#a3a3a3")
            self.adminUsernameEntry.configure(font=font10)
            self.adminUsernameEntry.configure(foreground="#000000")
            self.adminUsernameEntry.configure(insertbackground="black")

            self.adminPasswordEntry = tk.Entry(adminScreen)
            self.adminPasswordEntry.place(relx=0.384, rely=0.54, height=27, relwidth=0.362)
            self.adminPasswordEntry.configure(background="#D9D9D9")
            self.adminPasswordEntry.configure(disabledforeground="#a3a3a3")
            self.adminPasswordEntry.configure(font=font10)
            self.adminPasswordEntry.configure(foreground="#000000")
            self.adminPasswordEntry.configure(insertbackground="black")
            self.adminPasswordEntry.configure(show="*")

            self.clearAdminUsername = tk.Button(adminScreen)
            self.clearAdminUsername.place(relx=0.803, rely=0.347, height=38, width=66)
            self.clearAdminUsername.configure(activebackground="#ececec")
            self.clearAdminUsername.configure(activeforeground="#000000")
            self.clearAdminUsername.configure(background="#2E2E2E")
            self.clearAdminUsername.configure(disabledforeground="#a3a3a3")
            self.clearAdminUsername.configure(font=font10)
            self.clearAdminUsername.configure(foreground="#FFFFFF")
            self.clearAdminUsername.configure(highlightbackground="#d9d9d9")
            self.clearAdminUsername.configure(highlightcolor="black")
            self.clearAdminUsername.configure(pady="0")
            self.clearAdminUsername.configure(text='''Clear''')
            self.clearAdminUsername.configure(command=clearUsername)

            self.clearAdminPassword = tk.Button(adminScreen)
            self.clearAdminPassword.place(relx=0.803, rely=0.527, height=38, width=66)
            self.clearAdminPassword.configure(activebackground="#ececec")
            self.clearAdminPassword.configure(activeforeground="#000000")
            self.clearAdminPassword.configure(background="#2E2E2E")
            self.clearAdminPassword.configure(disabledforeground="#a3a3a3")
            self.clearAdminPassword.configure(font=font10)
            self.clearAdminPassword.configure(foreground="#FFFFFF")
            self.clearAdminPassword.configure(highlightbackground="#d9d9d9")
            self.clearAdminPassword.configure(highlightcolor="black")
            self.clearAdminPassword.configure(pady="0")
            self.clearAdminPassword.configure(text='''Clear''')
            self.clearAdminPassword.configure(command=clearPassword)

            self.adminLoginBtn = tk.Button(adminScreen)
            self.adminLoginBtn.place(relx=0.452, rely=0.848, height=38, width=80)
            self.adminLoginBtn.configure(activebackground="#ececec")
            self.adminLoginBtn.configure(activeforeground="#000000")
            self.adminLoginBtn.configure(background="#2E2E2E")
            self.adminLoginBtn.configure(disabledforeground="#a3a3a3")
            self.adminLoginBtn.configure(font=font10)
            self.adminLoginBtn.configure(foreground="#FFFFFF")
            self.adminLoginBtn.configure(highlightbackground="#d9d9d9")
            self.adminLoginBtn.configure(highlightcolor="black")
            self.adminLoginBtn.configure(pady="0")
            self.adminLoginBtn.configure(text='''Login''')
            self.adminLoginBtn.configure(command=administratorLogin)

            self.loginMessage = tk.Message(adminScreen)
            self.loginMessage.place(relx=0.096, rely=0.694, relheight=0.111, relwidth=0.795)
            self.loginMessage.configure(background="#1B1B1B")
            self.loginMessage.configure(font=font10)
            self.loginMessage.configure(foreground="#1B1B1B")
            self.loginMessage.configure(highlightbackground="#d9d9d9")
            self.loginMessage.configure(highlightcolor="black")
            self.loginMessage.configure(text='''Login Success''')
            self.loginMessage.configure(width=580)

            adminScreen.mainloop()

        top.geometry("1367x696+-9+0")
        top.minsize(120, 1)
        top.maxsize(1370, 749)
        top.resizable(0, 0)
        top.focus_force()
        top.title("SMART ABSENSI - FAHMI")
        top.configure(background="#1B1B1B")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.Title = tk.Message(top)
        self.Title.place(relx=-0.007, rely=0.042, relheight=0.134, relwidth=1.005)
        self.Title.configure(background="#2E2E2E")
        self.Title.configure(font="-family {SF Pro Display} -size 36 -weight bold")
        self.Title.configure(foreground="#FFFFFF")
        self.Title.configure(highlightbackground="#D9D9D9")
        self.Title.configure(highlightcolor="black")
        self.Title.configure(text='''Smart Absensi Absen Dengan Face Recognition''')
        self.Title.configure(width=1374)

        self.Notification = tk.Label(top)
        self.Notification.configure(text="COBA PINTU OTOMATIS")
        self.Notification.configure(background="#008000")
        self.Notification.configure(foreground="#FFFFFF")
        self.Notification.configure(width=64, height=2)
        self.Notification.configure(font="-family {SF Pro Display} -size 16 -weight bold")
        self.Notification.place(x=92, y=430)

        self.takeImages = tk.Button(top)
        self.takeImages.place(relx=0.067, rely=0.818, height=38, width=133)
        self.takeImages.configure(activebackground="#ececec")
        self.takeImages.configure(activeforeground="#000000")
        self.takeImages.configure(background="#2E2E2E")
        self.takeImages.configure(disabledforeground="#a3a3a3")
        self.takeImages.configure(font=font10)
        self.takeImages.configure(foreground="#FFFFFF")
        self.takeImages.configure(highlightbackground="#d9d9d9")
        self.takeImages.configure(highlightcolor="black")
        self.takeImages.configure(pady="0")
        self.takeImages.configure(text='''Take Images''')
        self.takeImages.configure(command=takeImage)

        self.trainStudent = tk.Button(top)
        self.trainStudent.place(relx=0.205, rely=0.818, height=38, width=139)
        self.trainStudent.configure(activebackground="#ececec")
        self.trainStudent.configure(activeforeground="#000000")
        self.trainStudent.configure(background="#2E2E2E")
        self.trainStudent.configure(disabledforeground="#a3a3a3")
        self.trainStudent.configure(font=font11)
        self.trainStudent.configure(foreground="#FFFFFF")
        self.trainStudent.configure(highlightbackground="#d9d9d9")
        self.trainStudent.configure(highlightcolor="black")
        self.trainStudent.configure(pady="0")
        self.trainStudent.configure(text='''Train Student''')
        self.trainStudent.configure(command=trainImage)

        self.automaticAttendance = tk.Button(top)
        self.automaticAttendance.place(relx=0.344, rely=0.818, height=38, width=220)
        self.automaticAttendance.configure(activebackground="#ececec")
        self.automaticAttendance.configure(activeforeground="#000000")
        self.automaticAttendance.configure(background="#2E2E2E")
        self.automaticAttendance.configure(disabledforeground="#a3a3a3")
        self.automaticAttendance.configure(font=font11)
        self.automaticAttendance.configure(foreground="#FFFFFF")
        self.automaticAttendance.configure(highlightbackground="#d9d9d9")
        self.automaticAttendance.configure(highlightcolor="black")
        self.automaticAttendance.configure(pady="0")
        self.automaticAttendance.configure(text='''Automatic Attendance''')
        self.automaticAttendance.configure(command=autoAttendance)

        self.manualAttendance = tk.Button(top)
        self.manualAttendance.place(relx=0.541, rely=0.818, height=38, width=194)
        self.manualAttendance.configure(activebackground="#ececec")
        self.manualAttendance.configure(activeforeground="#000000")
        self.manualAttendance.configure(background="#2E2E2E")
        self.manualAttendance.configure(disabledforeground="#a3a3a3")
        self.manualAttendance.configure(font=font11)
        self.manualAttendance.configure(foreground="#FFFFFF")
        self.manualAttendance.configure(highlightbackground="#d9d9d9")
        self.manualAttendance.configure(highlightcolor="black")
        self.manualAttendance.configure(pady="0")
        self.manualAttendance.configure(text='''Manual Attendance''')
        self.manualAttendance.configure(command=manualAttendance)

        self.adminPanel = tk.Button(top)
        self.adminPanel.place(relx=0.797, rely=0.345, height=38, width=131)
        self.adminPanel.configure(activebackground="#ececec")
        self.adminPanel.configure(activeforeground="#000000")
        self.adminPanel.configure(background="#2E2E2E")
        self.adminPanel.configure(disabledforeground="#a3a3a3")
        self.adminPanel.configure(font=font11)
        self.adminPanel.configure(foreground="#FFFFFF")
        self.adminPanel.configure(highlightbackground="#d9d9d9")
        self.adminPanel.configure(highlightcolor="black")
        self.adminPanel.configure(pady="0")
        self.adminPanel.configure(text='''Admin Panel''')
        self.adminPanel.configure(command=adminPanel)

        self.authorDetails = tk.Message(top)
        self.authorDetails.place(relx=0.753, rely=0.46, relheight=0.407, relwidth=0.19)
        self.authorDetails.configure(background="#2E2E2E")
        self.authorDetails.configure(font=font12)
        self.authorDetails.configure(foreground="#ffffff")
        self.authorDetails.configure(highlightbackground="#d9d9d9")
        self.authorDetails.configure(highlightcolor="black")
        self.authorDetails.configure(justify='center')
        self.authorDetails.configure(text='''This software is designed by Rushil Choksi & Modification by FAHMI''')
        self.authorDetails.configure(width=260)


# Disini mulai reload module dari do_something
if __name__ == '__main__':
    vp_start_gui()
    size = 10000000   
    n_exec = 10
    for i in range(0, n_exec):
        out_list = list()
        do_something(size, out_list)
    
    print ("List processing complete.")
    end_time = time.time()
    print("serial time=", end_time - start_time)