import vk
import os
import shutil
import time

def add_to(label, img):
    if not os.path.isdir(f'{path}/{label}'):
        os.mkdir(f'{path}/{label}')

    shutil.copy(f'{path}/{img}', f'{path}/{label}/{img}')

token = '4cf7394e4cf7394e4cf7394e9d4c80132844cf74cf7394e2c93ee5405f6ac620e8375b7'
session = vk.Session(access_token=token)
vk_api = vk.API(session, v=5.131)
path = "boxes"
categories = ["female", "male"]
box_sex = ["packaging box", "boxcutter"]

def sex(id: str, img):
    sex = 0
    gor = vk_api.users.get(user_id=id, fields = "sex")
    if len(gor) != 0:
        sex = gor[0].get("sex")
    if sex == 1:
        add_to(categories[0], img)
    if sex == 2:
        add_to(categories[1], img)


def box(id: str, img):
    sex = 0
    if len(gor) != 0:
        sex = gor[0].get("sex")
    if sex == 1:
        add_to(categories[0], img)
    if sex == 2:
        add_to(categories[1], img)


items = os.listdir(path)
for item in items:
    end = item.find("_")
    if end != -1:
        add_to(item[:end], item)
