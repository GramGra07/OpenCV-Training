from PIL import Image, UnidentifiedImageError
import os

class ImageLoader:
    def __init__(self, path):
        self.path = os.path.join("img", path)  # Ensure proper path handling
        self.image = None

    def load_image(self):
        try:
            print(f"Attempting to load image from: {self.path}")
            self.image = Image.open(self.path)
            self.image.load()  # Force loading the image into memory
            print("Image loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.path}")
            self.image = None
        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file at {self.path}")
            self.image = None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.image = None

    def get_image(self):
        if self.image is None:
            print("Image not loaded yet. Loading now...")
            self.load_image()
        return self.image

    def save_image(self, save_path, format=None):
        if self.image is not None:
            try:
                print(f"Saving image to: {save_path}")
                self.image.save(save_path, format=format)
                print("Image saved successfully.")
            except Exception as e:
                print(f"Error saving image: {e}")
        else:
            print("No image loaded to save.")
def getAlpha():
    loader = ImageLoader('alpha.jpg')
    loader.load_image()
    image = loader.get_image()
    return image
def getBoxes():
    loader = ImageLoader('tresBox.jpg')
    loader.load_image()
    image = loader.get_image()
    return image
