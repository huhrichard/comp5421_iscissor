import Tkinter as tk

def myfunction(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1)
    canvas.old_coords = x, y

root = tk.Tk()

canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()
canvas.old_coords = None

root.bind('<Motion>', myfunction)
root.mainloop()