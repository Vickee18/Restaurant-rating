from django.core.serializers import python
from django.shortcuts import render
from .models import hotel,review



# Create your views here.
def index(request):
    h=hotel.objects.all()
    return render(request,'application/index.html',{'hot':h})

def predict(request):
    import os
    import cv2
    f=""
    n=request.POST['choice']
    print("Choice:"+n)
    import numpy as np
    from tensorflow.keras.models import load_model, model_from_json
    from tensorflow.keras.preprocessing.image import load_img,img_to_array
    from tensorflow.keras.preprocessing import image
    model = model_from_json(open("pr_model/model_arch.json", "r").read())
    model.load_weights('pr_model/model.h5')
    # model = load_model('static\Fer2013.h5')
    face_haar_cascade = cv2.CascadeClassifier('pr_model/haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)

    while cap.isOpened():
        res,frame=cap.read()

        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6),0:int(width)]

        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        lable_color = (10, 10, 255)
        lable = "Emotion Detection"
        lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
        textX = int((res.shape[1] - lable_dimension[0]) / 2)
        textY = int((res.shape[0] + lable_dimension[1]) / 2)
        cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS)
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image )
        try:
            for (x,y, w, h) in faces:
                cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
                roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
                roi_gray=cv2.resize(roi_gray,(48,48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis = 0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                emotion_prediction = emotion_detection[max_index]
                f=emotion_prediction
                cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0,textY+22+5), FONT,0.7, lable_color,2)
                lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
                violation_text_dimension = cv2.getTextSize(lable_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
                violation_x_axis = int(res.shape[1]- violation_text_dimension[0])
                cv2.putText(res, lable_violation, (violation_x_axis,textY+22+5), FONT,0.7, lable_color,2)
        except :
            pass
        frame[0:int(height/6),0:int(width)] =res
        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f)
            if(f=='happy'):
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=5)
                s.save()
            elif(f=='surprise'):
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=4)
                s.save()
            elif(f=='neutral'):
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=3)
                s.save()
            elif(f=='sad'):
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=2)
                s.save()
            elif(f=='angry'):
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=1)
                s.save()
            else:
                h=hotel.objects.get(name=n)
                print(h.name)
                s=review.objects.create(hid=hotel.objects.get(id=h.id),rating=0)
                s.save()

            break



    cap.release()
    cv2.destroyAllWindows
    h=hotel.objects.all()
    return render(request,'application/index.html',{'hot':h})

def display(request):
    d=review.objects.all()
    return render(request,'application/predict.html',{'con':d})


def review_rat(request):
    from django.db.models import Avg,Sum
    hot=hotel.objects.all()
    for i in hot:
        m=review.objects.filter(hid=i).count()
        r=review.objects.filter(hid=i).aggregate(Sum('rating'))
        print(float(r['rating__sum']))
        u=hotel.objects.filter(id=i.id).update(avrg=(r['rating__sum']/m),tot=m) 
        #u.save()   
        print(m)
    return render(request,'application/avg.html',{'con':hotel.objects.all().order_by('-avrg')})
