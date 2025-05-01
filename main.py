import time
from openCV import *


def main():
    try:
        grey()
        print("Executed: grey()")
    except Exception as e:
        print(f"Error in grey(): {e}")
    time.sleep(1)
    try:
        ycrcb()
        print("Executed: ycrcb()")
    except Exception as e:
        print(f"Error in ycrcb(): {e}")
    time.sleep(1)
    try:
        hls()
        print("Executed: hls()")
    except Exception as e:
        print(f"Error in hls(): {e}")
    time.sleep(1)
    try:
        hsv()
        print("Executed: hsv()")
    except Exception as e:
        print(f"Error in hsv(): {e}")
    time.sleep(1)
    try:
        lab()
        print("Executed: lab()")
    except Exception as e:
        print(f"Error in lab(): {e}")
    time.sleep(1)
    try:
        submat()
        print("Executed: submat()")
    except Exception as e:
        print(f"Error in submat(): {e}")
    time.sleep(1)

    try:
        submatArea()
        print("Executed: submatArea()")
    except Exception as e:
        print(f"Error in submatArea(): {e}")
    time.sleep(1)

    try:
        blueColorThresh()
        print("Executed: blueColorThresh()")
    except Exception as e:
        print(f"Error in blueColorThresh(): {e}")
    time.sleep(1)

    try:
        yellowColorThresh()
        print("Executed: yellowColorThresh()")
    except Exception as e:
        print(f"Error in yellowColorThresh(): {e}")
    time.sleep(1)

    try:
        redColorThresh()
        print("Executed: redColorThresh()")
    except Exception as e:
        print(f"Error in redColorThresh(): {e}")
    time.sleep(1)

    try:
        contoursMax()
        print("Executed: contoursMax()")
    except Exception as e:
        print(f"Error in contoursMax(): {e}")
    time.sleep(1)

    try:
        practicalExample()
        print("Executed: practicalExample()")
    except Exception as e:
        print(f"Error in practicalExample(): {e}")
    time.sleep(1)

    try:
        findSamplesP1(0)
        print("Executed: findSamplesP1(0)")
    except Exception as e:
        print(f"Error in findSamplesP1(0): {e}")
    time.sleep(1)

    try:
        findSamplesP1(1)
        print("Executed: findSamplesP1(1)")
    except Exception as e:
        print(f"Error in findSamplesP1(1): {e}")
    time.sleep(1)

    try:
        findSamplesP2(0)
        print("Executed: findSamplesP2(0)")
    except Exception as e:
        print(f"Error in findSamplesP2(0): {e}")
    time.sleep(1)

    try:
        findSamplesP2(1)
        print("Executed: findSamplesP2(1)")
    except Exception as e:
        print(f"Error in findSamplesP2(1): {e}")
    time.sleep(1)

    try:
        findSamplesP3(0)
        print("Executed: findSamplesP3(0)")
    except Exception as e:
        print(f"Error in findSamplesP3(0): {e}")
    time.sleep(1)

    try:
        findSamplesP3(1)
        print("Executed: findSamplesP3(1)")
    except Exception as e:
        print(f"Error in findSamplesP3(1): {e}")
    time.sleep(1)

    try:
        findSamplesProblematic(0)
        print("Executed: findSamplesProblematic(0)")
    except Exception as e:
        print(f"Error in findSamplesProblematic(0): {e}")
    time.sleep(1)

    try:
        findSamplesProblematic(1)
        print("Executed: findSamplesProblematic(1)")
    except Exception as e:
        print(f"Error in findSamplesProblematic(1): {e}")
    time.sleep(1)

    # for i in range(2):
    i = 0
    for j in range(5):
        try:
            findSamplesProblematicZOOM(i, j)
            print(f"Executed: findSamplesProblematicZOOM({i}, {j})")
        except Exception as e:
            print(f"Error in findSamplesProblematicZOOM({i}, {j}): {e}")
        time.sleep(1)


if __name__ == '__main__':
    main()
