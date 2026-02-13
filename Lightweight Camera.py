import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def Camera():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None, None, None, None, None, None, None, None, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    bluecx = bluecy = orangecx = orangecy = yellowcx = yellowcy = None

    if True: # blue
        # blue colour range
        lower_blue = np.array([85, 100, 50])
        upper_blue = np.array([140, 255, 255])

        # blue mask
        maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

        # find boundaries
        bluecontours, _ = cv2.findContours(maskblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bluecx, bluecy = None, None
        bluex_min, bluey_min = float('inf'), float('inf')
        bluex_max, bluey_max = 0, 0

        for contour in bluecontours:
            area = cv2.contourArea(contour)
            if area > 500: # ignore small noise
                bluex, bluey, bluew, blueh = cv2.boundingRect(contour)
                bluex_min = min(bluex_min, bluex)
                bluey_min = min(bluey_min, bluey)
                bluex_max = max(bluex_max, bluex + bluew)
                bluey_max = max(bluey_max, bluey + blueh)

        if bluex_min < bluex_max and bluey_min < bluey_max:
            cv2.rectangle(frame, (bluex_min, bluey_min), (bluex_max, bluey_max), (255, 0, 0), 2)
            bluecx = (bluex_min + bluex_max) // 2
            bluecy = (bluey_min + bluey_max) // 2

    if True: # orange
        # orange colour range
        lower_orange = np.array([0, 180, 180])
        upper_orange = np.array([20, 255, 255])

        # orange mask
        maskorange = cv2.inRange(hsv, lower_orange, upper_orange)

        # find boundaries
        orangecontours, _ = cv2.findContours(maskorange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        orangecx, orangecy = None, None
        orangex_min, orangey_min = float('inf'), float('inf')
        orangex_max, orangey_max = 0, 0

        for contour in orangecontours:
            area = cv2.contourArea(contour)
            if area > 300: # ignore small noise
                orangex, orangey, orangew, orangeh = cv2.boundingRect(contour)
                orangex_min = min(orangex_min, orangex)
                orangey_min = min(orangey_min, orangey)
                orangex_max = max(orangex_max, orangex + orangew)
                orangey_max = max(orangey_max, orangey + orangeh)

        if orangex_min < orangex_max and orangey_min < orangey_max:
            cv2.rectangle(frame, (orangex_min, orangey_min), (orangex_max, orangey_max), (0, 165, 255), 2)
            orangecx = (orangex_min + orangex_max) // 2
            orangecy = (orangey_min + orangey_max) // 2

    if True: # yellow
        # yellow colour range
        lower_yellow = np.array([15, 90, 125])
        upper_yellow = np.array([30, 255, 255])

        # yellow mask
        maskyellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # find boundaries
        yellowcontours, _ = cv2.findContours(maskyellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yellowcx, yellowcy = None, None
        yellowx_min, yellowy_min = float('inf'), float('inf')
        yellowx_max, yellowy_max = 0, 0

        for contour in yellowcontours:
            area = cv2.contourArea(contour)
            if area > 500: # ignore small noise
                yellowx, yellowy, yelloww, yellowh = cv2.boundingRect(contour)
                yellowx_min = min(yellowx_min, yellowx)
                yellowy_min = min(yellowy_min, yellowy)
                yellowx_max = max(yellowx_max, yellowx + yelloww)
                yellowy_max = max(yellowy_max, yellowy + yellowh)

        if yellowx_min < yellowx_max and yellowy_min < yellowy_max:
            cv2.rectangle(frame, (yellowx_min, yellowy_min), (yellowx_max, yellowy_max), (0, 255, 255), 2)
            yellowcx = (yellowx_min + yellowx_max) // 2
            yellowcy = (yellowy_min + yellowy_max) // 2

    return bluecx, bluecy, frame, maskblue, orangecx, orangecy, maskorange, yellowcx, yellowcy, maskyellow

def GetPosition(minX, maxX, minY, colourX, colourY):
    if colourX is not None and colourY is not None:
        if maxX > colourX > minX and colourY > minY:
            return True
        else:
            return False
    else:
        return False
    
while True:
    bluecx, bluecy, frame, maskblue, orangecx, orangecy, maskorange, yellowcx, yellowcy, maskyellow = Camera()

    if frame is None or maskblue is None or maskyellow is None or maskorange is None:
        continue

    cv2.imshow("Original", frame)
    cv2.imshow("maskblue", maskblue)
    cv2.imshow("mask orange", maskorange)
    cv2.imshow("mask yellow", maskyellow)
    
    blueCentered = GetPosition(220,420,240,bluecx,bluecy)
    #print(blueCentered)

    yellowCentered = GetPosition(220,420,240,yellowcx,yellowcy)
    #print(yellowCentered)

    orangeCentered = GetPosition(270,370,240,orangecx,orangecy)
    #print(orangeCentered)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()