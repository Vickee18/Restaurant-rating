from django.contrib import admin

# Register your models here.
from .models import hotel,review

admin.site.register(hotel)
admin.site.register(review)
