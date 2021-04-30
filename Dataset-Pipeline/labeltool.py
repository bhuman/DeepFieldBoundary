"""
This script manages field boundary labels.
A dataset is represented by a directory containing PNG images and a CSV file.
A row in the CSV file consists of five columns:
    - The path to the image relative to the directory
    - The y coordinate of the field boundary at the left image border (0 = upper image border, 1 = lower image border)
    - The x coordinate of the 'peak' (0 = left image border, 1 = right image border)
    - The y coordinate of the 'peak' (0 = upper image border, 1 = lower image border)
    - The y coordinate of the field boundary at the right image border (0 = upper image border, 1 = lower image border)

The labeling process works as follows:
    - Call this script and pass it the path to a directory which contains PNG images (e.g. exported by "log saveImages")
    - The script will start at the first unlabeled image
    - The right and left arrow keys will go to the next / previous *unlabeled* image (as long as there are still some)
    - The down and up arrow keys will go to the next / previous image (regardless of existing labels)
    - The f key will label the image as "field boundary is above the image" (i.e. the complete image is field or objects on the field)
    - The v key will label the image as "field boundary is below the image" (i.e. there is no field in the image)
    - The space key is an alias for the right arrow key
        - The idea is that the left thumb is on the space key, the index finger on v and the middle finger on f,
          such that the right hand can remain at the mouse for rapid labeling
    - The backspace key deletes the label from the image
    - Add up to two lines by clicking two points on a line
        - Adding another line will remove the previous ones
"""

import argparse
import csv
import os
import pathlib
import tkinter as tk

from PIL import Image, ImageTk
from typing import Dict, Tuple

Label = Tuple[float, float, float, float]
LabelDict = Dict[str, Label]


def load_labels(path: str) -> LabelDict:
    result = {}
    try:
        with open(path, newline='') as label_file:
            reader = csv.reader(label_file)
            for row in reader:
                result[row[0]] = tuple([float(_) for _ in row[1:]])
    except FileNotFoundError:
        pass
    return result


def save_labels(path: str, labels: LabelDict):
    with open(path, 'w', newline='') as label_file:
        writer = csv.writer(label_file)
        writer.writerows([(image, y1, x2, y2, y3) for image, (y1, x2, y2, y3) in labels.items()])


class LabelApplication(tk.Frame):
    def __init__(self, directory, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.directory = directory
        self.CANVAS_WIDTH, self.CANVAS_HEIGHT = 640, 480
        self.image_canvas = tk.Canvas(self, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,
                                      highlightthickness=0, borderwidth=0, cursor='cross')

        self.image_canvas.pack(side='top')
        self.image_canvas.bind('<Button-1>', self.on_mouse_click)
        self.image_canvas.bind('<Button-3>', lambda event: self.on_reset())
        self.master.bind('<Configure>', self.resize)
        self.master.bind('<space>', lambda event: self.next_image(skip_labeled=True))
        self.master.bind('<Left>', lambda event: self.next_image(skip_labeled=True, sign=-1))
        self.master.bind('<Right>', lambda event: self.next_image(skip_labeled=True))
        self.master.bind('<Up>', lambda event: self.next_image(sign=-1))
        self.master.bind('<Down>', lambda event: self.next_image())
        self.master.bind('<BackSpace>', lambda event: self.on_reset())
        self.master.bind('f', lambda event: self.set_label((0, -1, -1, 0)))
        self.master.bind('v', lambda event: self.set_label((1, -1, -1, 1)))
        self.master.bind('e', lambda event: self.exclude_image())
        self.master.bind('h', lambda event: self.toggle_progress)

        # Load data.
        self.directory = directory
        self.images = sorted([_ for _ in os.listdir(self.directory) if _.endswith('.png')])
        self.label_path = os.path.join(self.directory, 'labels.csv')
        self.labels = load_labels(self.label_path)

        self.first_point = None

        # Proceed to first unlabeled image.
        self.current_image_index = len(self.images) - 1
        self.current_image_original = None
        self.current_image = None
        self.current_label = None
        self.show_progress = False
        self.next_image(skip_labeled=True)

    def save_label(self, to_file=True):
        if self.current_label is not None:
            self.labels[self.images[self.current_image_index]] = self.current_label
        else:
            self.labels.pop(self.images[self.current_image_index], None)
        if to_file:
            save_labels(self.label_path, self.labels)

    def next_image(self, skip_labeled=False, sign=1):
        start_index = self.current_image_index
        break_next = False
        while True:
            self.current_image_index += sign
            self.current_image_index %= len(self.images)
            if not skip_labeled or self.images[self.current_image_index] not in self.labels or break_next:
                break
            if self.current_image_index == start_index:
                break_next = True

        self.current_image_original = Image.open(os.path.join(self.directory, self.images[self.current_image_index]))
        self.current_image = ImageTk.PhotoImage(self.current_image_original.resize((self.CANVAS_WIDTH, self.CANVAS_HEIGHT), Image.ANTIALIAS))

        self.first_point = None
        try:
            self.current_label = self.labels[self.images[self.current_image_index]]
        except KeyError:
            self.current_label = None
        self.update_canvas()

    def update_canvas(self):
        self.image_canvas.delete('all')
        self.image_canvas.create_image(0, 0, image=self.current_image, anchor='nw')
        if self.current_label is not None:
            if self.current_label[1] == -1:
                self.image_canvas.create_line(0, self.current_label[0] * (self.CANVAS_HEIGHT - 1), self.CANVAS_WIDTH - 1,
                                              self.current_label[3] * (self.CANVAS_HEIGHT - 1), fill='red')
            else:
                self.image_canvas.create_line(0, self.current_label[0] * (self.CANVAS_HEIGHT - 1),
                                              self.current_label[1] * (self.CANVAS_WIDTH - 1), self.current_label[2] * (self.CANVAS_HEIGHT - 1), fill='red')
                self.image_canvas.create_line(self.current_label[1] * (self.CANVAS_WIDTH - 1),
                                              self.current_label[2] * (self.CANVAS_HEIGHT - 1), self.CANVAS_WIDTH - 1,
                                              self.current_label[3] * (self.CANVAS_HEIGHT - 1), fill='red')
        if self.show_progress:
            self.image_canvas.create_text(self.CANVAS_WIDTH//20, self.CANVAS_HEIGHT//20, fill="coral1", font="Times 10 italic bold", text=f'{len(self.labels)} / {len(self.images)}')

    def on_mouse_click(self, event):
        if self.first_point is not None:
            dx = event.widget.canvasx(event.x) - self.first_point[0]
            if dx == 0:
                dx = 1
            m = (event.widget.canvasy(event.y) - self.first_point[1]) / dx
            y0 = ((0 - self.first_point[0]) * m + self.first_point[1]) / self.CANVAS_HEIGHT
            y1 = ((self.CANVAS_WIDTH - 1 - self.first_point[0]) * m + self.first_point[1]) / self.CANVAS_HEIGHT

            if self.current_label is not None and self.current_label[1] == -1:
                m *= (self.CANVAS_WIDTH / self.CANVAS_HEIGHT)
                m_other = self.current_label[3] - self.current_label[0]
                x_intersection = (self.current_label[0] - y0) / (m - m_other)
                if x_intersection < 0 or x_intersection > 1:
                    self.first_point = None
                    return
                self.current_label = (max(y0, self.current_label[0]), x_intersection,
                                      m * x_intersection + y0, max(y1, self.current_label[3]))
            else:
                self.current_label = (y0, -1, -1, y1)
            self.save_label()
            self.update_canvas()

            self.first_point = None
        else:
            self.first_point = (event.widget.canvasx(event.x), event.widget.canvasy(event.y))

    def on_reset(self):
        self.first_point = None
        self.set_label(None)

    def set_label(self, label):
        self.current_label = label
        self.save_label()
        self.update_canvas()

    def exclude_image(self):
        self.on_reset()

        pathlib.Path(f'{self.directory}/excluded').mkdir(parents=True, exist_ok=True)
        os.rename(f'{self.directory}/{self.images[self.current_image_index]}', f'{self.directory}/excluded/{self.images[self.current_image_index]}')
        del self.images[self.current_image_index]
        self.current_image_index -= 1

        self.next_image(skip_labeled=True)

    def toggle_progress(self):
        self.show_progress = not self.show_progress
        self.update_canvas()

    def resize(self, event):
        if self.CANVAS_WIDTH == event.width and self.CANVAS_HEIGHT == event.height:
            return

        self.CANVAS_WIDTH = event.width
        self.CANVAS_HEIGHT = event.height
        self.master.config(width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.image_canvas.config(width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)

        self.current_image = ImageTk.PhotoImage(self.current_image_original.resize((self.CANVAS_WIDTH, self.CANVAS_HEIGHT)))
        self.update_canvas()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script manages field boundary labels.')
    parser.add_argument('directory', nargs='?', default='.')
    args = parser.parse_args()

    root = tk.Tk()
    root.title('Field Boundary Labeling')
    app = LabelApplication(args.directory, master=root)
    app.mainloop()
