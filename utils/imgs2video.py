import cv2
import os


def create_video_from_images(folder_path, output_path, fps=20, size=(1335, 684)):
    """
    Create a video from all images in the specified folder using OpenCV.

    :param folder_path: Path to the folder containing images
    :param output_path: Path where the output video will be saved
    :param fps: Frames per second for the output video
    :param size: Size of the output video frame (width, height)
    """
    # Get list of image files in the folder
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    # Sort the files to ensure consistent ordering
    image_files.sort()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for image_file in image_files:
        file_path = os.path.join(folder_path, image_file)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Error reading image: {file_path}")
            continue

        # Resize image
        img = cv2.resize(img, size)

        # Write the frame to the video
        out.write(img)

    # Release everything when job is finished
    out.release()

    print(f"Video created successfully: {output_path}")


if __name__ == "__main__":
    folder_path = "/Users/parteeksj/Desktop/GIF-CREATE"
    output_path = "output_video_20fps.mp4"
    create_video_from_images(folder_path, output_path)
