import mediapipe as mp
import numpy as np
import cv2
import time
import math
import pyautogui


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

button_size = 40
space_bar_size = 200
gap = 5
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2
thickness = 1
typing_text = ''
typing_text_F = ''
click_state = 0
click_location = 0
on_focus = False
back_space_state = 0
enter_state = 0
real_length = None
hand_length = None
rate = None

key_value_1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
key_value_2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
key_value_3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm', 'E']
key_value_4 = ['space', 'back']

#키보드를 반투명하게 해주는 함수
def add_white_rectangle(frame, x, y, w, h):
    sub_img = frame[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    frame[y:y+h, x:x+w] = res

    return frame

#키보드를 그리는 함수
def draw_keyboard(frame):
    frame = add_white_rectangle(frame, 98, 100, button_size*10+gap*9, button_size)
    for x in range(10):
        frame = add_white_rectangle(frame, x*(button_size+gap)+98, 100+button_size+gap, button_size, button_size)
        cv2.putText(frame, f'{key_value_1[x]}', (x*(button_size+gap)+108, 125+button_size+gap), font, fontScale, fontColor, lineType)
    for x in range(9):
        frame = add_white_rectangle(frame, x*(button_size+gap)+button_size//2+98, 100+button_size*2+gap*2, button_size, button_size)
        cv2.putText(frame, f'{key_value_2[x]}', (x*(button_size+gap)+button_size//2+108, 125+button_size*2+gap*2), font, fontScale, fontColor, lineType)
    for x in range(8):
        frame = add_white_rectangle(frame, x*(button_size+gap)+button_size+98, 100+button_size*3+gap*3, button_size, button_size)
        cv2.putText(frame, f'{key_value_3[x]}', (x*(button_size+gap)+button_size+108, 125+button_size*3+gap*3), font, fontScale, fontColor, lineType)
    for x in range(2):
        frame = add_white_rectangle(frame, x*(space_bar_size+gap+20)+98, 100+button_size*4+gap*4, button_size*5+gap*4, button_size)
        cv2.putText(frame, f'{key_value_4[x]}', (x*(space_bar_size+gap+20)+98+button_size*2, 100+button_size*4+gap*4+25), font, fontScale, fontColor, lineType)
    return frame


#손의 좌표를 계산하여 클릭을 구분하는 함수
def calculating_hand_org():
  global click_state, real_length, hand_length, rate
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      # #1
      # if abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*frame_width) < 25:
      #   if abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*frame_height) < 25:
      #     click_state = 1
      #   else:
      #       click_state = 0
      # else:
      #     click_state = 0

      # #2
      # real_length = math.sqrt(pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*frame_width)) + pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*frame_height)))
      # if real_length < 25:
      #   click_state = 1
      # else:
      #   click_state = 0

      #3
      real_length = math.sqrt(pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*frame_width), 2) + pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*frame_height), 2))
      hand_length = math.sqrt(pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*frame_width - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x*frame_width), 2) + pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*frame_height - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y*frame_height), 2))
      rate = hand_length / real_length
      if rate > 3:
          click_state = 1
      else:
        click_state = 0

#클릭이 되었을 때 해당 키가 무엇인지 계산하는 함수
def typing_calculate(typing_text_F, hand_x, hand_y):
  global click_location, typing_text, back_space_state, enter_state
  if hand_y >= 100+button_size+gap and hand_y < 100+button_size*2+gap:
     for x in range(10):
        if hand_x > x*(button_size+gap)+108 and hand_x < x*(button_size+gap)+108+button_size:
          typing_text = key_value_1[x]
          return typing_text_F + key_value_1[x]
  elif hand_y >= 100+button_size*2+gap*2 and hand_y < 100+button_size*3+gap*2:
     for x in range(9):
        if hand_x > x*(button_size+gap)+button_size//2+98 and hand_x < x*(button_size+gap)+button_size//2+98+button_size:
          typing_text = key_value_2[x]
          return typing_text_F + key_value_2[x]
  elif hand_y >= 100+button_size*3+gap*3 and hand_y < 100+button_size*4+gap*3:
     for x in range(8):
        if hand_x > x*(button_size+gap)+button_size+98 and hand_x < x*(button_size+gap)+button_size+98+button_size:
          if key_value_3[x] == 'E':
            enter_state = 1
            print(enter_state)
          else:
            typing_text = key_value_3[x]
            return typing_text_F + key_value_3[x]
  elif hand_y >= 100+button_size*4+gap*4 and hand_y < 100+button_size*5+gap*4:
     for x in range(2):
        if hand_x > x*(space_bar_size+gap+20)+98 and hand_x < x*(space_bar_size+gap+20)+98+button_size*5+gap*4:
          if key_value_4[x] == 'space':
            return typing_text_F + ' '
          elif key_value_4[x] == 'back':
            return typing_text_F[:-1]
  else:
    print("other space")
    return None

#크롬을 켜서 YouTube를 자동으로 열어주는 함수
def open_chrome_and_navigate_to_youtube():
    try:
        # 윈도우 키를 누르고 놓음
        pyautogui.press('win')
        time.sleep(0.5)

        # "크롬"을 검색하여 실행
        pyautogui.typewrite('chrome')
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(1)

        # 주소창에 YouTube 주소 입력
        pyautogui.typewrite('youtube.com')
        time.sleep(0.5)
        pyautogui.press('enter')
    except Exception as e:
        print("크롬을 실행하고 YouTube로 이동할 수 없습니다.")

#디스코드를 자동으로 열어주는 함수
def open_discord():
    try:
        # 윈도우 키를 누르고 놓음
        pyautogui.press('win')
        time.sleep(0.5)

        # "Discord"을 검색하여 실행
        pyautogui.typewrite('Discord')
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(1)
    except Exception as e:
        print("디스코드를 실행할 수 없습니다.")

#윈도우 키를 자동으로 눌러주는 함수
def open_windows_search():
    try:
        # 윈도우 키를 누르고 놓음
        pyautogui.press('win')
        time.sleep(0.5)
        
    except Exception as e:
        print("검색 창을 열 수 없습니다.")


cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("카메라를 찾을 수 없습니다.")
      # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용
      continue
    frame = cv2.flip(frame, 1)
      

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame_height, frame_width, _ = frame.shape

    # 이미지에 손 주석을 그림
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hand_x = 0
    hand_y = 0
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width
        hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height
        
    frame = draw_keyboard(frame)
    calculating_hand_org()
    cnt = 0
    if click_state == 1:
        if on_focus == False:
          start_time = time.time()
          on_focus = True
        elif on_focus == True:
          current_time = time.time()
          if current_time - start_time >= 1.5:
            if typing_calculate(typing_text_F, hand_x, hand_y) != None:
              if enter_state == 1:
                if typing_text == 'e':
                  #검색창 실행
                  open_windows_search()
                  enter_state = 0
                elif typing_text == 'y':
                  # 크롬 실행 및 YouTube로 이동
                  open_chrome_and_navigate_to_youtube()
                  enter_state = 0
                elif typing_text == 'd':
                  open_discord()
              else:
                typing_text_F = typing_calculate(typing_text_F, hand_x, hand_y)
            else:
              pass
            on_focus = False
             
    cv2.putText(frame, f'{typing_text_F}', (108, 125), font, fontScale, fontColor, lineType)
    cv2.imshow('frame', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()