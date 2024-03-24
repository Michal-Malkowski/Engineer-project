import tkinter as tk
from StartPage import StartPage

WIDTH = 100
HEIGHT = 100

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


if __name__ == "__main__":
    myapp = App()
    myapp.master.title("App")
    myapp.master.minsize(WIDTH, HEIGHT)
    myapp.mainloop()
