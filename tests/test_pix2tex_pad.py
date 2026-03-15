from PIL import Image
import numpy as np
import cv2
import traceback

print("Testing Pix2Tex padding function manually...")

def pad(img: Image, divable: int = 32) -> Image:
    threshold = 128
    data = np.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255-data[..., -1]).astype(np.uint8)
    data = (data-data.min())/(data.max()-data.min())*255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255*(data < threshold).astype(np.uint8)
    else:
        gray = 255*(data > threshold).astype(np.uint8)
        data = 255-data

    # This is where it fails if gray is empty or invalid
    print(f"Gray shape: {gray.shape}, dtype: {gray.dtype}")
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    
    if coords is None:
        print("WARNING: findNonZero returned None (no text found).")
        return img # Fallback
        
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    im = Image.fromarray(rect).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded

try:
    print("\nTesting with pure white image (no text)...")
    img_white = Image.new('RGB', (100, 30), color=(255, 255, 255))
    result = pad(img_white)
    print("Success!")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    traceback.print_exc()

try:
    print("\nTesting with some black text...")
    img_text = Image.new('RGB', (100, 30), color=(255, 255, 255))
    # Add a black pixel to simulate text
    img_text.putpixel((50, 15), (0, 0, 0))
    result = pad(img_text)
    print("Success!")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

