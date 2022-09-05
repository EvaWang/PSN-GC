import pickle
import csv
import json
from io import StringIO
import math

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch

def dump_json(json_object, file_path):
    with open(file_path, "w", encoding='utf-8') as outfile:
        json.dump(json_object, outfile, ensure_ascii=False)

def read_json(file_path):
    f = open(file_path, "r")
    data = json.load(f)
    f.close()
    return data

def dump_csv(fieldnames, data, file_path):
    with open(file_path, mode='w', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def read_csv(file_path, delimiter=','):
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        data = csv_file.read()
        data = data.replace('\x00','XX')
        csv_reader = csv.DictReader(StringIO(data), delimiter = delimiter)
        # csv_reader = csv.DictReader(csv_file)
        # print(csv_reader)
        for row in csv_reader:
            data_list.append(row)
    return data_list

def dump_data(file_path, data):
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()

def read_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def create_img_by_font(char, font_name, img_size, font_color=255, bg_color=0):
    ft = ImageFont.truetype(font_name, img_size-2)
    w, h = ft.getsize(char)
    offset_x, offset_y = ft.getoffset(char)
    # print(f"{offset_x}, {offset_y}")
    # print(f"{w}, {h}")
    
    h = h-offset_y
    w = w-offset_x

    img = Image.new("L", (img_size,img_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw.text(((img_size-w)/2-offset_x, (img_size-h)/2-offset_y), char, font=ft, fill=font_color)

    if img.size!=(img_size,img_size):
        print(img_size)
        print(char)
        raise Exception("wrong size")

    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
# def read