# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# import mediapipe as mp
# from PIL import Image,ImageDraw,ImageFont
# mp_face_mesh=mp.solutions.face_mesh
# face_mesh=mp_face_mesh.FaceMesh(static_image_mode=False,
#                       max_num_faces=1,
#                       min_detection_confidence=0.5,
#                       min_tracking_confidence=0.5)
# model=load_model('fer2013_mini_XCEPTION.102-0.66.hdf5',compile=False)#here we are loading pretrained model
# print(model)
# # i.e already trained on emotion classification problem on fer 2013 dataset

# #emotion labels

# emotion_labels={
#     0:'angry',
#     1:'disgust',
#     2:'fear',
#     3:'happy',
#     4:'sad',
#     5:'surprise',
#     6:'neutral',
# }
# emoji_dict = {
#     'angry': '😠',
#     'disgust': '🤢',
#     'fear': '😨',
#     'happy': '😊',
#     'sad': '😢',
#     'surprise': '😲',
#     'neutral': '😐'
# }

# #initialize webcam
# cap=cv2.VideoCapture(0)


# #font for emoji overlay(using pil)
# # font=ImageFont.truetype("Apple color Emoji.ttc",80)
# font = ImageFont.truetype("seguiemj.ttf", 40)
# while cap.isOpened():
#     ret,frame=cap.read()
#     # print(ret,frame)
#     if not ret:
#         break
#     #convert bgr to rgb
#     rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     # print(rgb_frame)
#     results=face_mesh.process(rgb_frame)
#     print(f'mediapipe ke results:{results}')
#     # print(results.multi_face_landmarks)

#     if results.multi_face_landmarks:
    
#         for face_landmarks in results.multi_face_landmarks:
#             print(f'face ka landmarks is:{face_landmarks}')
#             print(face_landmarks.landmark[0])
#             h,w,_=frame.shape
#             print(f'ye frame ka shape hai:{frame.shape}')
#             x_min=w
#             x_max=0
#             y_min=h
#             y_max=0
#             for landmark in face_landmarks.landmark:
#                 x,y=int(landmark.x*w),int(landmark.y*h)
#                 print(f'ye x hai:{x} and ye y hai {y}')
#                 x_min = min(x_min, x)
#                 print(x_min)
#                 y_min = min(y_min, y)
#                 print(y_min)
#                 x_max = max(x_max, x)
#                 print(x_max)
#                 y_max = max(y_max, y)
#                 print(y_max)
#                 padding = 20
#                 x_min = max(0, x_min - padding)#max(0,1270-20)#max(0,1250)
#                 y_min = max(0, y_min - padding)
#                 x_max = min(w, x_max + padding)
#                 y_max = min(h, y_max + padding)
#                 face_roi=frame[y_min:y_max,x_min:x_max]
#                 if face_roi.size!=0:
#                     gray_face=cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY)#converted face roi to grayscale
#                     resized_face=cv2.resize(gray_face,(64,64))#converted gray scale to 64*64 pixels
#                     normalized_face=resized_face/255.0
#                     reshaped_face = np.reshape(normalized_face,(1, 64, 64, 1))
#                     emotions_probabilities=model.predict(reshaped_face)
#                     print(emotions_probabilities)
#                     emotion_index=np.argmax(emotions_probabilities)
#                     emotion=emotion_labels[emotion_index]
#                     print(emotion)
#                     emoji=emoji_dict.get(emotion,'😂')
#                     print(emoji)
#                     pil_img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#                     draw=ImageDraw.Draw(pil_img)
#                     #calculate posiition for emoji
#                     emoji_x=x_min+(x_max-x_min)//2-40
#                     emoji_y=y_min-100
#                     #draw emoji
#                     draw.text((emoji_x,emoji_y),emoji,font=font,embedded_color=True)
#                     #convert back to open cv format
#                     frame=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
#                     cv2.putText(frame,emotion,(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
#     cv2.imshow('emoji mirror',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()







import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Load Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Load pretrained emotion detection model
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Emotion labels and emoji map
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral',
}
emoji_dict = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

# Load emoji font
font = ImageFont.truetype("seguiemj.ttf", 40)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                continue

            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (64, 64))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

            emotions_probabilities = model.predict(reshaped_face, verbose=0)
            emotion_index = np.argmax(emotions_probabilities)
            emotion = emotion_labels[emotion_index]
            emoji = emoji_dict.get(emotion, '😐')

            # Draw emoji using PIL
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            emoji_x = x_min + (x_max - x_min) // 2 - 20
            emoji_y = y_min - 60
            draw.text((emoji_x, emoji_y), emoji, font=font, embedded_color=True)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, emotion, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emoji Mirror', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
