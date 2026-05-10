# parking_lpr_local_ui_v12.py
# Raspberry Pi 停車場車牌辨識（Plate ROI 版 / Safe LCD）
# USB Camera + Tesseract + LCD1602 I2C 0x3F + OpenCV Local UI
# Author: Matt

import cv2
import pytesseract
import re
import time
import csv
import os
from datetime import datetime
from RPLCD.i2c import CharLCD


# =========================
# LCD1602 I2C (0x3F)
# =========================
lcd = CharLCD(
    'PCF8574',
    0x3F,
    cols=16,
    rows=2,
    charmap='A00',
    auto_linebreaks=True
)

# =========================
# 系統設定
# =========================
CAMERA_INDEX = 0
LOG_DIR = "logs"
CSV_FILE = os.path.join(LOG_DIR, "parking_log.csv")
SAVE_IMAGE = True
DETECT_INTERVAL = 1.0
DUPLICATE_COOLDOWN = 8

WINDOW_NAME = "Parking LPR Local UI"
WINDOW_W = 960
WINDOW_H = 640

# =========================
# 初始化
# =========================
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "plate_number", "image_file"])

last_seen = {}
latest_plate = "----"
last_plate = "----"
latest_time = "--:--:--"
status_text = "Scanning"
last_box = None

# =========================
# LCD (Safe / Anti-I2C Flood)
# =========================
last_lcd_line1 = ""
last_lcd_line2 = ""
last_lcd_update = 0
LCD_MIN_INTERVAL = 0.25   # 250ms，避免 I2C 連續寫爆


def lcd_show(line1="", line2="", force=False):
    global last_lcd_line1, last_lcd_line2, last_lcd_update

    line1 = str(line1)[:16].ljust(16)
    line2 = str(line2)[:16].ljust(16)

    now = time.time()

    # 內容沒變就不刷新
    if not force and line1 == last_lcd_line1 and line2 == last_lcd_line2:
        return

    # 避免太短時間連續寫入
    if not force and (now - last_lcd_update) < LCD_MIN_INTERVAL:
        return

    try:
        # 不再用 clear()，直接覆蓋寫入
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1)
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2)

        last_lcd_line1 = line1
        last_lcd_line2 = line2
        last_lcd_update = now

    except OSError:
        # LCD 暫時失聯時避免整個主程式炸掉
        pass


def lcd_boot():
    lcd_show("Parking LPR", "System Ready", force=True)
    time.sleep(2)
    lcd_show("Scanning...", "Please Wait", force=True)


# =========================
# OCR Helpers
# =========================
def preprocess_plate(plate_roi):
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def normalize_plate(text):
    text = text.strip().upper()
    text = re.sub(r'[^A-Z0-9-]', '', text)

    # 常見 OCR 誤判修正
    text = text.replace('O', '0') if re.match(r'.*\d.*', text) else text
    text = text.replace('I', '1')
    text = text.replace('S', '5')
    text = text.replace('B', '8') if re.match(r'^\d', text) else text

    # 自動補 dash
    if '-' not in text and len(text) in (6, 7, 8):
        if len(text) == 7:
            text = text[:3] + '-' + text[3:]
        elif len(text) == 6:
            text = text[:2] + '-' + text[2:]
        elif len(text) == 8:
            text = text[:4] + '-' + text[4:]

    return text


# =========================
# Plate ROI Detection
# =========================
def find_plate_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            # 台灣車牌大概比例
            if 2.0 < ratio < 5.5 and w > 120 and h > 40:
                roi = frame[y:y+h, x:x+w]
                return roi, (x, y, w, h)

    return None, None


def extract_plate_text(frame):
    roi, box = find_plate_roi(frame)
    if roi is None:
        return None, None

    processed = preprocess_plate(roi)

    text = pytesseract.image_to_string(
        processed,
        config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    )

    text = normalize_plate(text)

    if 6 <= len(text) <= 9:
        return text, box

    return None, box


# =========================
# Logging
# =========================
def save_log(plate, frame):
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = now.strftime("%Y%m%d_%H%M%S")

    image_file = ""
    if SAVE_IMAGE:
        image_file = os.path.join(LOG_DIR, f"{file_ts}_{plate}.jpg")
        cv2.imwrite(image_file, frame)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, plate, image_file])

    print(f"[LOG] {ts} | {plate} | {image_file}")


def is_duplicate(plate):
    now = time.time()
    if plate in last_seen and (now - last_seen[plate] < DUPLICATE_COOLDOWN):
        return True
    last_seen[plate] = now
    return False


# =========================
# UI
# =========================
def draw_ui(frame):
    ui = cv2.resize(frame, (WINDOW_W, WINDOW_H))

    if last_box:
        x, y, w, h = last_box
        sx = WINDOW_W / frame.shape[1]
        sy = WINDOW_H / frame.shape[0]
        x, y, w, h = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
        cv2.rectangle(ui, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.rectangle(ui, (0, 0), (WINDOW_W, 50), (40, 40, 40), -1)
    cv2.putText(ui, "Parking LPR System", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(ui, (0, WINDOW_H - 140), (WINDOW_W, WINDOW_H), (30, 30, 30), -1)

    cv2.putText(ui, f"Plate : {latest_plate}", (20, WINDOW_H - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(ui, f"Time  : {latest_time}", (20, WINDOW_H - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(ui, f"Status: {status_text}", (20, WINDOW_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(ui, f"Last  : {last_plate}", (520, WINDOW_H - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    return ui


# =========================
# Main
# =========================
def main():
    global latest_plate, last_plate, latest_time, status_text, last_box

    lcd_boot()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        lcd_show("Camera Error", "Check Device", force=True)
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Parking LPR Local UI v1.2 started")
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        status_text = "Scanning"

        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now

            plate, box = extract_plate_text(frame)
            last_box = box

            if plate and not is_duplicate(plate):
                last_plate = latest_plate
                latest_plate = plate
                latest_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_text = "Detected"

                print(f"[DETECTED] {plate}")
                lcd_show("Plate Number:", plate, force=True)
                save_log(plate, frame)

                time.sleep(5)
                lcd_show("Scanning...", "Please Wait", force=True)

        ui = draw_ui(frame)
        cv2.imshow(WINDOW_NAME, ui)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
