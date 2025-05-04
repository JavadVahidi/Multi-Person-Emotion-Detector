import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import speech_recognition as sr
import keyboard
import threading

# مسیر فونت فارسی
font_path = "arial.ttf"
try:
    font = ImageFont.truetype(font_path, 32)
except OSError:
    print("Error: Cannot load default font. Falling back to OpenCV font.")
    font = None

# بارگذاری مدل تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# تعریف لیست کامل احساسات
emotion_map = {
    "happy": "خوشحال", "sad": "غمگین", "angry": "عصبانی", "neutral": "خنثی",
    "surprise": "متعجب", "fear": "ترسیده", "disgust": "متنفر", "love": "عاشق",
    "hope": "امیدوار", "shame": "شرمنده", "embarrassment": "خجالت‌زده", "jealousy": "حسود",
    "pride": "مغرور", "guilt": "گناهکار", "joy": "شاد", "melancholy": "دلتنگ",
    "excitement": "هیجان‌زده", "despair": "ناامید", "awe": "متعجب با احترام",
    "confusion": "گیج", "loneliness": "تنها", "gratitude": "قدردان", "nostalgia": "نوستالژیک",
    "euphoria": "سرخوش", "frustration": "ناامید و خشمگین", "irritation": "رنجیده",
    "curiosity": "کنجکاو", "sympathy": "همدل", "relief": "آرامش‌یافته", "anxiety": "مضطرب",
    "contentment": "راضی", "trust": "مطمئن", "anticipation": "منتظر", "self-confidence": "با اعتماد به نفس",
    "self-doubt": "شک به خود", "admiration": "تحسین‌گر", "passion": "پرشور", "acceptance": "پذیرفته",
    "regret": "پشیمان", "comfort": "آرام", "bravery": "شجاع", "calmness": "آرامش",
    "respect": "احترام‌گذار", "determination": "مصمم", "envy": "حسادت", "surprise-delight": "متعجب و خوشحال",
    "emboldened": "جرأت‌مند", "enthusiasm": "مشتاق"
}

# ذخیره اطلاعات چهره و احساسات
people_emotions = {}

# تابع تشخیص چهره و احساسات
def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            except:
                emotion = "neutral"

            person_id = f"person_{i+1}"
            if person_id not in people_emotions:
                people_emotions[person_id] = {key: 0 for key in emotion_map.keys()}
            people_emotions[person_id][emotion] += 1

            emotion_farsi = emotion_map.get(emotion, "نامشخص")
            reshaped_text = arabic_reshaper.reshape(emotion_farsi)
            bidi_text = get_display(reshaped_text)

            if font:
                bbox = draw.textbbox((0, 0), bidi_text, font=font)
                text_width = bbox[2] - bbox[0]
                x_position = x + (w // 2) - (text_width // 2)
                draw.text((x_position, y - 30), bidi_text, font=font, fill=(255, 255, 255))
            else:
                cv2.putText(frame, bidi_text[::-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Multi-Person Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    for person_id, emotion_counts in people_emotions.items():
        total_counts = sum(emotion_counts.values())
        emotion_percentages = {emotion_map[key]: (value / total_counts) * 100 for key, value in emotion_counts.items()}

        df = pd.DataFrame(list(emotion_percentages.items()), columns=["احساس", "درصد"])
        df = df.sort_values(by="درصد", ascending=False)

        filename = f"{person_id}.xlsx"
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"فایل اکسل برای {person_id} با نام '{filename}' ذخیره شد.")

# تابع تشخیص گفتار
def recognize_speech():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 1

    with sr.Microphone() as source:
        welcome_text = arabic_reshaper.reshape("لطفاً صحبت کنید (برای خروج 'q' را فشار دهید)...")
        print(get_display(welcome_text))

        recognizer.adjust_for_ambient_noise(source, duration=2)

        while True:
            if keyboard.is_pressed("q"):
                exit_text = arabic_reshaper.reshape("برنامه بسته شد.")
                print(get_display(exit_text))
                break

            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=7)
                text = recognizer.recognize_google(audio, language="fa-IR")

                reshaped_text = arabic_reshaper.reshape(text)
                correct_text = get_display(reshaped_text)

                print(correct_text)
                with open("speech_text.txt", "a", encoding="utf-8") as file:
                    file.write(text + "\n")

            except sr.UnknownValueError:
                error_text = arabic_reshaper.reshape("متأسفم، نتوانستم چیزی بفهمم.")
                print(get_display(error_text))
            except sr.RequestError:
                error_text = arabic_reshaper.reshape("خطا در ارتباط با سرویس Google Speech Recognition.")
                print(get_display(error_text))
            except Exception as e:
                print(f"خطای پیش‌بینی‌نشده: {e}")

# اجرای همزمان دو برنامه
face_thread = threading.Thread(target=detect_faces)
speech_thread = threading.Thread(target=recognize_speech)

face_thread.start()
speech_thread.start()

face_thread.join()
speech_thread.join()
