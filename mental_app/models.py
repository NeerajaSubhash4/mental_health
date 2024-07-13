from django.db import models

class Registration(models.Model):
    name=models.CharField(max_length=100)
    address=models.CharField(max_length=100)
    phone=models.BigIntegerField()
    email=models.EmailField()

class Therapist(models.Model):
    name=models.CharField(max_length=100)
    address=models.CharField(max_length=100)
    phone=models.BigIntegerField()
    email=models.EmailField()
    qualification=models.CharField(max_length=100)
    experience=models.CharField(max_length=100)

class Booking(models.Model):
    regid=models.ForeignKey(Registration,on_delete=models.CASCADE)
    therapistid=models.ForeignKey(Therapist,on_delete=models.CASCADE)
    bookingdate=models.DateField()
    status=models.CharField(max_length=100)

class Question(models.Model):
    question=models.CharField(max_length=500)

class Responsemaster(models.Model):
    bookingid=models.ForeignKey(Booking,on_delete=models.CASCADE)
    responsedate=models.DateField(auto_now_add=True)
    level=models.CharField(max_length=100,default='')

class Responsechild(models.Model):
    respid=models.ForeignKey(Responsemaster,on_delete=models.CASCADE)
    question=models.CharField(max_length=500)
    response=models.CharField(max_length=50)

class Survey(models.Model):
    bookingid=models.ForeignKey(Booking,on_delete=models.CASCADE)
    sdate=models.DateField(auto_now_add=True)
    positive=models.IntegerField(default=0)
    negative=models.IntegerField(default=0)
    neutral=models.IntegerField(default=0)
    sentiment=models.CharField(max_length=50,blank=True)

class Questionnaire(models.Model):
    surveyid=models.ForeignKey(Survey,on_delete=models.CASCADE)
    question=models.CharField(max_length=500)
    answer=models.CharField(max_length=500)