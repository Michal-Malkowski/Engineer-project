import tkinter as tk
from tkinter.filedialog import askopenfilename
import cv2
import threading
from skimage import io

from Processing import Processing

alpha = 0
beta = 255

class StartPage(tk.Frame):
    def __init__(self,master):
        self.alpha = 0
        self.beta = 255
        tk.Frame.__init__(self, master, bg='azure')
        tk.Label(self, text='Mobile Microscope', font=('bold', 48), bg='azure').pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Start Camera", font=('bold', 24), bg='gainsboro',
                  command=lambda: self.URL()).pack(pady=10)
        tk.Button(self, text="Camera settings", font=('bold', 24), bg='gainsboro',
                  command=lambda: master.switch_frame(Settings)).pack(pady=10)
        tk.Button(self, text="Image processing", font=('bold', 24), bg='gainsboro',
                  command=lambda: master.switch_frame(ImageProcessing)).pack(pady=10)
        tk.Button(self, text="Exit", font=('bold', 24), bg='gainsboro',
                  command=lambda: exit()).pack(pady=10)


    def URL(self):
        root = tk.Tk()
        root.title('URL')
        label = tk.Label(root, text='Podaj adres IP telefonu:', font=('bold', 14)).pack()
        text_box = tk.Text(root, width=40, height=1)
        text_box.pack()
        button = tk.Button(root, text='Zatwierdź', font=('bold', 14),
                             command=lambda: self.startCamera("http://"+text_box.get('1.0', 'end')+"/video")).pack()


    def startCamera(self, url):
        t_camera = threading.Thread(target= self.getVideo(url))
        t_camera.run()

    def getVideo(self, url):
        Processing.loadModel(Processing)
        outlineRealTime = False
        counter = 0
        img_number = 1
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        global alpha
        global beta
        while (True):
            counter += 1
            if counter > 100000:
                counter = 0
            video, frame = cap.read()

            img_resize = cv2.resize(frame, (512, 512))

            cv2.normalize(img_resize, img_resize, alpha, beta, cv2.NORM_MINMAX)

            if frame is not None and not outlineRealTime:
                cv2.imshow("Frame", img_resize)
            elif frame is not None and outlineRealTime and counter%2 == 0:
                newFrame = Processing.drawOutline(Processing, frame, True)
                text = "Area of skin lesion: " + str(Processing.getArea(Processing)) + "%"
                cv2.putText(newFrame, text, (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Frame", newFrame)
            q = cv2.waitKey(1)
            if q == ord(" "):
                break
            if q == ord("p"):
                cv2.imwrite("image" + str(img_number) + ".jpg", frame)
                img_number += 1
            if q == ord("o"):
                outlineRealTime = not outlineRealTime

        cap.release()
        cv2.destroyAllWindows()

class Settings(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='azure')
        tk.Label(self, text='Mobile Microscope', font=('bold', 48), bg='azure').pack(side="top", fill="x", pady=10)
        tk.Label(self, text='Brightness', font=(24), bg='azure').pack(side="top", fill="x", pady=10)
        self.w1 = tk.Scale(self, from_=0, to=100, bg='azure', orient='horizontal')
        self.w1.set(50)
        self.w1.pack()
        tk.Label(self, text='Contrast', font=(24), bg='azure').pack(side="top", fill="x", pady=10)
        self.w2 = tk.Scale(self, from_=0, to=100, bg='azure', orient='horizontal')
        self.w2.set(50)
        self.w2.pack()
        tk.Button(self, text="Return to main menu", font=('bold', 24), bg='gainsboro',
                  command=lambda: self.backToPreviousPage(master)).pack(pady=10)

    def backToPreviousPage(self, master):
        global alpha
        global beta
        brightness = self.w1.get() - 50
        contrast = self.w2.get() - 50
        alpha += brightness * 2
        beta += brightness * 2
        alpha += contrast * 2
        beta -= contrast * 2
        master.switch_frame(StartPage)



class ImageProcessing(tk.Frame):
    def __init__(self, master):
        self.img_number = 4
        Processing.loadModel(Processing)
        tk.Frame.__init__(self, master, bg='azure')
        tk.Label(self, text='Mobile Microscope', font=('bold', 48), bg='azure').pack(side="top", fill="x", pady=10)
        self.loadLabel = tk.Label(self, text='', font=(24), bg='azure')
        options = ["None", "Segmentation", "Outline", "Mean (imagej function)", "Median (imagej function)"]
        variable = tk.StringVar()
        variable.set(options[0])
        w = tk.OptionMenu(self, variable, *options, command=lambda: self.setOption)
        w.config(font=(24), bg='azure')
        w.pack(pady=10)
        tk.Button(self, text="Load image", font=('bold', 24), bg='gainsboro',
                  command=lambda: self.loadImage()).pack(pady=10)
        tk.Button(self, text="Open image", font=('bold', 24), bg='gainsboro',
                  command=lambda: self.showImage(variable.get())).pack(pady=10)
        tk.Button(self, text="Return to main menu", font=('bold', 24), bg='gainsboro',
                  command=lambda: master.switch_frame(StartPage)).pack(pady=10)


    def loadImage(self):
        self.filename = askopenfilename()
        str = 'File ' + self.filename + ' loaded'
        self.loadLabel.config(text=str)
        self.loadLabel.pack(side="top", fill="x", pady=10)
        self.img = io.imread(self.filename)

    def showImage(self, option):
        match option:
            case "None":
                self.drawResize(self.img)
                cv2.imshow('image', self.img)
            case "Segmentation":
                self.drawSegmentation(self.img)
                cv2.imshow('image', self.img)
            case "Outline":
                self.drawOutline(self.img)
                cv2.imshow('image', self.img)
            case "Mean (imagej function)":
                self.drawMean(self.filename)
                cv2.imshow('image', self.img)
            case "Median (imagej function)":
                self.drawMedian(self.filename)
                cv2.imshow('image', self.img)

        self.img_number += 1
        self.img = io.imread(self.filename)



    def setOption(self, selectedOption):
        self.option = selectedOption

    def drawResize(self, img):
        self.img = cv2.cvtColor(Processing.drawResize(Processing, img), cv2.COLOR_BGR2RGB)

    def drawSegmentation(self, img):
        self.img = Processing.drawSegmentation(Processing, img, False)

    def drawOutline(self, img):
        self.img = Processing.drawOutline(Processing, img, False)

    def drawMean(self, filename):
        self.img = Processing.drawMean(Processing, filename)

    def drawMedian(self, filename):
        self.img = Processing.drawMedian(Processing, filename)