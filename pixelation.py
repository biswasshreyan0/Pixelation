from PIL import Image

def pixelate_image(image_path, output_path, pixel_size):
    # Open the image
    image = Image.open(image_path)

    # Get the original dimensions
    width, height = image.size

    # Resize the image to a smaller size (smaller pixel size), then back to original size
    image = image.resize(
        (width // pixel_size, height // pixel_size),
        resample=Image.NEAREST
    )

    # Resize it back to the original dimensions
    image = image.resize((width, height), Image.NEAREST)

    # Save the pixelated image
    image.save(output_path)
    print(f"Pixelated image saved as {output_path}")

# Example usage
input_image = "Downloads/image_file.jpg"  # Replace with image file path
pixel_size = [5, 10, 15, 20, 25, 30]  # Size of the pixels (adjust as needed)

for size in pixel_size:
    # Path to save the pixelated image
    output_image = "Downloads/Pixelated_image_" + str(size) + ".jpg" 
    pixelate_image(input_image, output_image, size)
