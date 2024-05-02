import cv2
import pytesseract
from datetime import datetime

# Path to Tesseract executable (change this if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the cascade classifier for license plate detection
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

cap = cv2.VideoCapture(0)
frameWidth = 1000   # Frame Width
frameHeight = 480   # Frame Height
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("Number Plate", imgRoi)

            # Perform text extraction using Tesseract OCR
            extracted_text = pytesseract.image_to_string(imgRoi)

            # Get the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Print the extracted text and time to the console
            print("Extracted Text:")
            print(extracted_text)
            print("Time:", current_time)

            # Save the extracted text and time to a new file
            file_name = f"extracted_text_{count}.txt"
            with open(file_name, "w") as text_file:
                text_file.write("Time: " + current_time + "\n")
                text_file.write("Extracted Text:\n" + extracted_text)

            count += 1

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f".\IMAGES\{str(count)}.jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
