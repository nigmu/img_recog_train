import cv2


def extract_frames(video_path, interval, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize frame counter
    frame_counter = 0

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # If the frame was not successfully read then we have reached the end of the video
        if not ret:
            break

        # If this is the right frame to save based on the interval (in seconds)
        if frame_counter % (30 * interval) == 0:
            # Resize the frame to 196x28 pixels
            resized_frame = cv2.resize(frame, (196, 28))

            # Save the frame to disk
            cv2.imwrite(
                f"{output_path}/frame{frame_counter // (30 * interval)}.png",
                resized_frame,
            )

        # Increment our frame counter
        frame_counter += 1

    # Release the video file
    video.release()


# Call the function with your video file path, interval time (in seconds), and output directory path
extract_frames(
    "videos\\op_video\\output.mp4",
    5,
    "images\\op_image",
)
