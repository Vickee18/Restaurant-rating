from django.db import models
from django.db.models import Avg
# Create your models here.
class hotel(models.Model):
    name=models.CharField(max_length=100)
    contact=models.CharField(max_length=100)
    avrg=models.FloatField(null=True)
    tot=models.IntegerField(null=True)

    

    def __str__(self):
        
        return str(self.name)

    
    def averagereview(self):
        print("hello")
        r= review.objects.filter(hotel=self).aggregate(avarage=Avg('review__rating'))
        avg=0
        if r["avarage"] is not None:
            avg=float(r["avarage"])
        print("Average"+avg)
        return avg

class review(models.Model):
    hid=models.ForeignKey(hotel,on_delete=models.CASCADE)
    rating=models.IntegerField(null=True)

    

    def __str__(self):
        return str(self.hid)


