from PIL import Image
import os
import uuid

IMG_PATH = "Boulderei2"
OUTPUT_PATH = "Boulderei2/out"

def crop(img, wstrips, hstrips, fltr, new_img_path):
    w, h = img.size

    ctr = 0
    for i in range(hstrips):
        for j in range(wstrips):
            left = (w // wstrips) * j
            top = (h // hstrips) * i
            right = (w // wstrips) * (j + 1)
            bottom = (h // hstrips) * (i + 1)
            new_img = img.crop((left, top, right, bottom))

            if ctr in fltr:
                new_img.save(os.path.join(new_img_path, str(uuid.uuid4()) + ".jpeg"), "JPEG")
                #new_img.show()

            ctr += 1

img_files = [f for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f))]
for img_file in img_files:
    img = Image.open(os.path.join(IMG_PATH, img_file))
    w, h = img.size

    if w > h:
        img = img.rotate(270, expand=True)
        w, h = img.size

    crop(img, 3, 2, range(6), OUTPUT_PATH)
    crop(img, 3, 3, [1, 4, 7], OUTPUT_PATH)
    crop(img.crop((0, h//6, w, h//6 * 5)), 2, 2, range(4), OUTPUT_PATH)
    #img.save(os.path.join(OUTPUT_PATH, str(uuid.uuid4()) + ".jpeg"), "JPEG")