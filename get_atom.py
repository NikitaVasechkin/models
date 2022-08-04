import vk
import requests
import os
import time


def download_url(url: str, likes: int, folder: str, id: str, timestamp: int):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print(f'Folder {folder} successfully created')

    path_to_file = f'{folder}/{likes}_{id}_{timestamp}.jpg'
    if not os.path.exists(path_to_file):
        img_data = requests.get(url).content
        with open(path_to_file, 'wb') as handler:
            handler.write(img_data)
    return 0


def get_id(text: str):
    start = text.find("[")
    end = text.find("|")
    prompt = text[start+1:end]
    if (start or end) == -1 or ("/" in prompt) or (":" in prompt) or ("." in prompt):
        return 0
    else:
        return prompt

session = vk.Session(access_token=token)
vk_api = vk.API(session, v=5.131)
folder = "./data"
batch = 100
cond = True
offset = 12400
offset_threshold = 12500


while (cond == True and offset <= offset_threshold):
    print(f'Current offset: {offset}', end='\r')
    doner = vk_api.wall.get(domain='iate_atomohod', count=batch, offset=offset)
    for times, elem in enumerate(doner.get("items")):
        if len(elem) == 0:
            cond = False
            print(f'Met false condition at object {offset+times}')
        if "Подписчик" in elem.get("text"):
            id = get_id(elem.get("text"))
            stamp = elem.get("date")
            attachments = elem.get("attachments")
            if attachments is not None:
                photo = attachments[0].get("photo")
                if photo is not None:
                    sizes = photo.get("sizes")
                    if sizes is not None:
                        url = sizes[-1].get("url")
                    else:
                        print(
                            f'Error: could not retrieve url at object {offset+times}')
                        break
            likes_dict = elem.get("likes")
            if likes_dict is not None:
                likes = likes_dict.get("count")
                if likes is not None:
                    download_url(url=url, likes=likes,
                                 folder=folder, id=id, timestamp=stamp)
                else:
                    print(
                        f'Error: could not retreive number of likes at object {offset+times}')
    offset += batch
