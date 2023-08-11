from django.contrib import admin


from django.urls import path,include
from . import views

urlpatterns = [
    
    path('',views.index,name='index'),
    path('predict/',views.predict,name='predict'),
    path('display/',views.display,name='display'),
    path('review_rat/',views.review_rat,name='review_rat')
]