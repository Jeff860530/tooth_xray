import numpy as np
import imutils
import glob
import cv2

if __name__ == "__main__":
        dirname = "001742"
        templete_image = [i for i in glob.iglob("Data/%s/*/*.PNG" % dirname)]
        target_image =  [i for i in glob.iglob("Data/%s/NN*.png" % dirname)]
        image_o = cv2.imread(target_image[0]) # image
        image = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)

        green = 0

        for index, tmp_img in enumerate(templete_image):
                template = cv2.imread(tmp_img) # template image
                
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                loc = False
                threshold = 0.8
                w, h = template.shape[::-1]
                
                for scale in np.linspace(0.1, 0.3, 20)[::-1]:
                        resized = imutils.resize(template, width = int(template.shape[1] * scale))
                        w, h = resized.shape[::-1]
                        res = cv2.matchTemplate(image,resized,cv2.TM_CCOEFF_NORMED)

                        loc = np.where( res >= threshold)
                        if len( list(zip(*loc[::-1])) ) > 0:
                                break

                if loc and len( list(zip(*loc[::-1])) ) > 0:
                        for pt in zip(*loc[::-1]):
                                cv2.rectangle(image_o, pt, (pt[0] + w, pt[1] + h), (0, green, 255), 2)

                                text = str(index + 1)
                                cv2.putText(image_o, text, (pt[0] + w - 100, pt[1] + h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, green, 255), 1, cv2.LINE_AA)
                                break
                green += 12

        cv2.namedWindow('Matched Template', cv2.WINDOW_NORMAL)
        cv2.imshow('Matched Template', image_o)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite( "Mapping Results.jpg", image_o );

