import sys
from PIL import Image, ImageDraw
import numpy as np
import cv2


def main():
    source_image = Image.open('example.png')
    source_width, source_height = source_image.size
    print('Image is {}x{}'.format(source_width, source_height))

    target_width = 200
    target_height = 200

    # Make image a reasonable size to work with. Using the source_height will
    # make sure it's just resized to the target_width
    source_image.thumbnail((target_width, source_height), Image.ANTIALIAS)

    # Find the faces and show us where they are
    faces = faces_from_pil_image(source_image)
    faces_found_image = draw_faces(source_image, faces)
    faces_found_image.show()

    # Get details about where the faces are so we can crop
    top_of_faces = top_face_top(faces)
    bottom_of_faces = bottom_face_bottom(faces)

    all_faces_height = bottom_of_faces - top_of_faces
    print('Faces are {} pixels high'.format(all_faces_height))

    if all_faces_height >= target_width:
        print('Faces take up more than the final image, you need better logic')
        exit_code = 1
    else:
        # Figure out where to crop and show the results
        # Fix the coordinates
        face_buffer = 0.5 * (target_height - all_faces_height)
        top_of_crop = int(top_of_faces - face_buffer)
        coords = (0, top_of_crop, target_width, top_of_crop + target_height)
        print('Cropping to', coords)
        final_image = source_image.crop(coords)
        final_image.show()
        exit_code = 0

    return exit_code


def faces_from_pil_image(pil_image):
    "Return a list of (x,y,w,h) tuples for faces detected in the PIL image"
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # Convert PIL image to OpenCV image format
    opencv_image = np.array(pil_image)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        opencv_image, scaleFactor=1.1, minNeighbors=5)

    return faces


def top_face_top(faces):
    coords = [f[1] for f in faces]
    # Top left corner is 0,0 so we need the min for highest face
    return min(coords)


def bottom_face_bottom(faces):
    # Top left corner is 0,0 so we need the max for lowest face. Also add the
    # height of the faces so that we get the bottom of it
    coords = [f[1] + f[3] for f in faces]
    return max(coords)


def draw_faces(image_, faces):
    "Draw a rectangle around each face discovered"
    image = image_.copy()
    drawable = ImageDraw.Draw(image)

    for x, y, w, h in faces:
        absolute_coords = (x, y, x + w, y + h)

        drawable.rectangle(absolute_coords)
    return image


if __name__ == '__main__':
    sys.exit(main())
