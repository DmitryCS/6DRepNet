import cv2
from math import cos, sin
import cv2, numpy as np
import tritonclient.grpc as grpcclient
from typing import Tuple
from sixdrepnet.regressor import SixDRepNet_Detector

def get_faces(img_raw):
    img_raw = cv2.resize(img_raw, (1280, 720))

    img = np.float32(img_raw)

    img -= (104, 117, 123)
    input_batch = np.expand_dims(img, axis=0)

    url = 'localhost'
    port = '8001'

    triton_model_name = 'face_detection'
    client = grpcclient.InferenceServerClient(url=f'{url}:{port}')
    config = client.get_model_config(triton_model_name, as_json=True)
    outputs_triton = []

    input_name = config['config']['input'][0]['name']
    for output in config['config']['output']:
        outputs_triton.append(grpcclient.InferRequestedOutput(output['name']))

    inputs = [grpcclient.InferInput(input_name, list(input_batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(input_batch)

    while True:
        try:
            results = client.infer(
                model_name=triton_model_name,
                inputs=inputs,
                outputs=outputs_triton,
                client_timeout=3,
                timeout=3,
                headers={},
                compression_algorithm=None)
            break   
        except:
            print(f'Failed infer {triton_model_name}. Try again.')
    confidences = results.as_numpy('confidences')
    condition = confidences > 0.1
    # landmarks = results.as_numpy('landmarks')[condition]
    locations = results.as_numpy('locations')[condition]
    return locations / np.array([1280, 720, 1280, 720], dtype=np.float32)

def get_angles(img_raw, infer='triton'):
    img_raw = resize_with_pad(img_raw, (224, 224))
    if infer == 'torch':
        model = SixDRepNet_Detector()
        pitch, yaw, roll = model.predict(img_raw)
        return roll, pitch, yaw
    elif infer == 'triton':
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        input_batch = np.expand_dims(img, axis=0)
        url = 'localhost'
        port = '8001'

        triton_model_name = 'head_pose_estimation'
        client = grpcclient.InferenceServerClient(url=f'{url}:{port}')
        config = client.get_model_config(triton_model_name, as_json=True)
        outputs_triton = []

        input_name = config['config']['input'][0]['name']
        for output in config['config']['output']:
            outputs_triton.append(grpcclient.InferRequestedOutput(output['name']))

        inputs = [grpcclient.InferInput(input_name, list(input_batch.shape), "UINT8")]
        inputs[0].set_data_from_numpy(input_batch)
        while True:
            try:
                results = client.infer(
                    model_name=triton_model_name,
                    inputs=inputs,
                    outputs=outputs_triton,
                    client_timeout=3,
                    timeout=3,
                    headers={},
                    compression_algorithm=None)
                break   
            except:
                print(f'Failed infer {triton_model_name}. Try again.')

        pred = results.as_numpy('output')
        pitch = pred[:, 0]
        yaw = pred[:, 1]
        roll = pred[:, 2]
        return roll, pitch, yaw
    
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def resize_with_pad(image: np.array, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])

    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new_shape)) / min(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,None,value=padding_color)
    return image

def draw_line(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def draw_text(img, yaw, pitch, roll, tdx=10, tdy=10):
    text = ''
    color = (0, 255, 255)
    threshold = 10
    if max(abs(yaw), abs(pitch)) < threshold:
        text = f'Forward {max(abs(yaw), abs(pitch)):.1f} < {threshold}'
        color = (255, 0, 0)
    elif abs(yaw) > abs(pitch):
        if yaw < -threshold:
            text = f'Looking Right {abs(yaw):.1f} > {threshold}'
        elif yaw > threshold:
            text = f'Looking Left {abs(yaw):.1f} > {threshold}'
    else:
        if pitch < -threshold:
            text = f'Looking Down {abs(pitch):.1f} > {threshold}'
        elif pitch > threshold:
            text = f'Looking Up {abs(pitch):.1f} > {threshold}'    
    cv2.putText(img, text, (tdx, tdy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2, lineType=2)


def draw_box(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
    return img
