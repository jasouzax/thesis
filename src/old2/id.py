import cv2

def list_camera_ids():
    """
    Tests potential camera indices and returns a list of working camera IDs.
    """
    available_camera_ids = []
    # Start checking from index 0, which is typically the default camera
    dev_port = 0 
    
    # Continue checking until a certain number of consecutive non-working ports are found
    # This helps in preventing an infinite loop if there are no cameras or a large gap in indices
    max_non_working_ports = 5 
    non_working_ports_count = 0

    while non_working_ports_count < max_non_working_ports:
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            print(f"Camera found at index: {dev_port}")
            available_camera_ids.append(dev_port)
            non_working_ports_count = 0  # Reset counter if a camera is found
            camera.release() # Release the camera after checking
        else:
            print(f"No camera found at index: {dev_port}")
            non_working_ports_count += 1
        dev_port += 1
        
    return available_camera_ids

if __name__ == "__main__":
    working_cameras = list_camera_ids()
    if working_cameras:
        print(f"\nSuccessfully found cameras at IDs: {working_cameras}")
    else:
        print("\nNo cameras found.")