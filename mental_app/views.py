import re
from django.shortcuts import render, redirect
from .models import Registration, Responsemaster, Therapist, Booking, Question,Responsechild, Questionnaire,Survey
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib import messages
import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
#from sklearn.metrics import roc_auc_score

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')

def retrieveTweet(data_url):

    tweets_data_path = data_url
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

             
def retrieveProcessedData(Pdata_url):
    sent = pd.read_excel(Pdata_url)
    for i in range(len(tweets_data)):
        if tweets_data[i]['id']==sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')          
            
            
def nbTrain():
    from sklearn.naive_bayes import MultinomialNB
    start_timenb = time.time()
    train_features = vectorizer.fit_transform(x)
    
    actual = y
    
    nb = MultinomialNB()
    nb.fit(train_features, [int(r) for r in y])
    
    test_features = vectorizer.transform(x)
    predictions = nb.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
    nbscore = format(metrics.auc(fpr, tpr))
    nbscore = float(nbscore)*100
    
    nb_matrix = confusion_matrix(actual, predictions)
    plt.figure()
    plot_confusion_matrix(nb_matrix, classes=[-1,0,1], title='Confusion matrix For NB classifier')
    
    print("\n")

    print("Naive Bayes  Accuracy : \n", nbscore,"%")
    print(" Completion Speed", round((time.time() - start_timenb),5))
    print()

def datree():
    from sklearn import tree
    start_timedt = time.time()
    train_featurestree = vectorizer.fit_transform(x)
    actual1 = y
    test_features1 = vectorizer.transform(x)
    dtree = tree.DecisionTreeClassifier()
    
    dtree = dtree.fit(train_featurestree, [int(r) for r in y])
    
    prediction1 = dtree.predict(test_features1)
    ddd, ttt, thresholds = metrics.roc_curve(actual1, prediction1, pos_label=1)
    dtreescore = format(metrics.auc(ddd, ttt))
    dtreescore = float(dtreescore)*100
    print("Decision tree Accuracy : \n", dtreescore, "%")
    print(" Completion Speed", round((time.time() - start_timedt),5))
    print()

def Tsvm():
    from sklearn.svm import SVC
    start_timesvm = time.time()
    train_featuressvm = vectorizer.fit_transform(x)
    actual2 = y
    test_features2 = vectorizer.transform(x)
    svc = SVC()
    
    svc = svc.fit(train_featuressvm, [int(r) for r in y])
    prediction2 = svc.predict(test_features2)
    sss, vvv, thresholds = metrics.roc_curve(actual2, prediction2, pos_label=1)
    svc = format(metrics.auc(sss, vvv))
    svc = float(svc)*100
    print("Support vector machine Accuracy : \n", svc, "%")
    print(" Completion Speed", round((time.time() - start_timesvm),5))
    print()

def knN():
    from sklearn.neighbors import KNeighborsClassifier
    start_timekn = time.time()
    train_featureskn = vectorizer.fit_transform(x)
    actual3 = y
    test_features3 = vectorizer.transform(x)
    kn = KNeighborsClassifier(n_neighbors=2)
    
    
    kn = kn.fit(train_featureskn, [int(i) for i in y])
    prediction3 = kn.predict(test_features3)
    kkk, nnn, thresholds = metrics.roc_curve(actual3, prediction3, pos_label=1)
    kn = format(metrics.auc(kkk, nnn))
    kn = float(kn)*100
    
    print("Kneighborsclassifier Accuracy : \n", kn, "%")
    print(" Completion Speed", round((time.time() - start_timekn),5))
    print()

def RanFo():
    from sklearn.ensemble import RandomForestClassifier
    start_timerf = time.time()
    train_featuresrf = vectorizer.fit_transform(x)
    actual4 = y
    test_features4 = vectorizer.transform(x)
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    
    
    rf = rf.fit(train_featuresrf, [int(i) for i in y])
    prediction4 = rf.predict(test_features4)
    rrr, fff, thresholds = metrics.roc_curve(actual4, prediction4, pos_label=1)
    kn = format(metrics.auc(rrr, fff))
    kn = float(kn)*100
    print("Random Forest Accuracy : \n", kn, "%")
    print(" Completion Speed", round((time.time() - start_timerf),5))
    print()
    print()


def runall():     
    retrieveTweet('data/tweetdata.txt')  
    retrieveProcessedData('processed_data/output.xlsx')
    nbTrain()
    datree()
    Tsvm()
    knN()
    RanFo()
    
def datreeINPUT(inputtweet):
    from sklearn import tree
    train_featurestree = vectorizer.fit_transform(x)
    dtree = tree.DecisionTreeClassifier()
    
    dtree = dtree.fit(train_featurestree, [int(r) for r in y])
    
    
    inputdtree= vectorizer.transform([inputtweet])
    predictt = dtree.predict(inputdtree)
    
    if predictt == 1:
        predictt = "Positive"
    elif predictt == 0:
        predictt = "Neutral"
    elif predictt == -1:
        predictt = "Negative"
    else:
        print("Nothing")
    
    print("\n*****************")
    print(predictt)
    print("*****************")
    return predictt

runall()


######################################################################
#
#
#                           COMMON
#
#
######################################################################
######################################################################
#                   LOAD INDEX PAGE
######################################################################


def index(request):
    """ 
        The function to load index page of the project. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    return render(request, 'index.html')
######################################################################
#                   LOAD LOGIN PAGE
######################################################################


def login(request):
    """ 
        The function to load index page of the project. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    if request.POST:
        # read the username and password given in UI
        email = request.POST['txtEmail']
        pwd = request.POST['txtPassword']
        # checking whether the client is admin

        # checking whether username and email exist in authenticate table
        user = authenticate(username=email, password=pwd)
        if user is None:
            #username or password is incorrect
            messages.info(request, 'Username or password incorrect')
        else:
            #username and password is correct
            user_data = User.objects.get(username=email)
            if user_data.is_superuser == 1:
                # if admin, goto admin interface
                return redirect("/adminhome")
            else:
                # if user, go to user interface
                if user_data.is_staff == 0:
                    request.session["email"] = email
                    r = Registration.objects.get(email=email)
                    request.session["id"] = r.id
                    request.session["name"] = r.name
                    return redirect("/patienthome")
                elif user_data.is_staff == 1:
                    request.session["email"] = email
                    r = Therapist.objects.get(email=email)
                    request.session["id"] = r.id
                    request.session["name"] = r.name
                    return redirect("/therapisthome")

    return render(request, 'login.html')
######################################################################
#                   LOAD REGISTRATION PAGE
######################################################################


def registration(request):
    """ 
        The function to load registration page of the project. 
        -------------------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    if request.POST:
        # read the values from UI
        name = request.POST['txtName']
        address = request.POST['txtAddress']
        phone = request.POST['txtContact']
        email = request.POST['txtEmail']
        pwd = request.POST['txtPassword']
        # check whether any duplicate entries occur
        user = authenticate(username=email, password=pwd)
        if user is None:
            # if no duplicate entries add the data to registration table
            try:
                r = Registration.objects.create(
                    name=name, address=address, phone=phone, email=email)
                r.save()
            except:
                messages.info(request, 'Sorry some error occured')
            else:
                # add the data to login table also
                try:
                    u = User.objects.create_user(
                        password=pwd, username=email, is_superuser=0, is_active=1, is_staff=0, email=email)
                    u.save()
                except:
                    messages.info(request, 'Sorry some error occured')
                else:
                    messages.info(request, 'Registration successfull')
        else:
            # duplicate entries occur and registration is not possible
            messages.info(request, 'User already registered')
    return render(request, 'registration.html')
######################################################################
#
#
#                           ADMIN
#
#
######################################################################
######################################################################
#                           LOAD ADMIN HOME PAGE
######################################################################


def adminhome(request):
    """ 
        The function to load admin page of the admin. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    return render(request, "adminhome.html")
######################################################################
#                   LOAD REGISTRATION PAGE
######################################################################


def admintherapist(request):
    """ 
        The function to add therapist. 
        -------------------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    if request.POST:
        # read the values from UI
        name = request.POST['txtName']
        address = request.POST['txtAddress']
        phone = request.POST['txtContact']
        qual = request.POST['txtQualification']
        exp = request.POST['txtExperience']
        email = request.POST['txtEmail']
        pwd = request.POST['txtPassword']
        # check whether any duplicate entries occur
        user = authenticate(username=email, password=pwd)
        if user is None:
            # if no duplicate entries add the data to registration table
            try:
                r = Therapist.objects.create(
                    name=name, address=address, phone=phone, email=email, qualification=qual, experience=exp)
                r.save()
            except:
                messages.info(request, 'Sorry some error occured')
            else:
                # add the data to login table also
                try:
                    u = User.objects.create_user(
                        password=pwd, username=email, is_superuser=0, is_active=1, is_staff=1, email=email)
                    u.save()
                except:
                    messages.info(request, 'Sorry some error occured')
                else:
                    messages.info(request, 'Registration successfull')
        else:
            # duplicate entries occur and registration is not possible
            messages.info(request, 'User already registered')
    # fetch and load all therapist from database
    therapist_data = Therapist.objects.all()
    return render(request, 'admintherapist.html', {"therapist": therapist_data})
######################################################################
#                   LOAD PATIENT PAGE
######################################################################


def adminpatient(request):
    """ 
        The function to view patients. 
        -------------------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    # fetch and load all therapist from database
    patient_data = Registration.objects.all()
    return render(request, 'adminpatient.html', {"patient": patient_data})
######################################################################
#
#
#                           THERAPIST
#
#
######################################################################
######################################################################
#                           LOAD DOCTOR HOME PAGE
######################################################################
def therapisthome(request):
    """ 
        The function to load index page of the therapist. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    return render(request, "therapisthome.html")
######################################################################
#                           LOAD DOCTOR BOOKING PAGE
######################################################################
def therapistbooking(request):
    """ 
        The function to load booking page of the therapist. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id=request.session["id"]
    rid=Therapist.objects.get(id=id)
    booking=Booking.objects.filter(therapistid=rid)
    return render(request, "therapistbooking.html",{"booking":booking})
######################################################################
#                           LOAD DOCTOR PATIENT PAGE
######################################################################
def therapistpatient(request):
    """ 
        The function to load patient page of the therapist. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id=request.GET.get("id")
    booking=Booking.objects.get(id=id)
    responsemaster=Responsemaster.objects.get(bookingid=id)
    print(responsemaster)
    rid=responsemaster.id
    print(rid)
    responsechild=Responsechild.objects.filter(respid=rid)
    survey=Survey.objects.get(bookingid=id)
    questionnaire=Questionnaire.objects.filter(surveyid=survey.id)
    return render(request, "therapistpatient.html",{"booking":booking,"responsemaster":responsemaster, "survey":survey,"questionnaire":questionnaire,"responsechild":responsechild})
######################################################################
#
#
#                           PATIENT
#
#
######################################################################
######################################################################
#                           LOAD PATIENT HOME PAGE
######################################################################


def patienthome(request):
    """ 
        The function to load index page of the patient. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id = request.session["id"]
    profile_data = Registration.objects.filter(id=id)
    return render(request, "patienthome.html", {"profile_data": profile_data})
######################################################################
#                           LOAD PATIENT QUESTION PAGE
######################################################################
def patientquestion(request):
    """ 
        The function to load question page of the patient. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id = request.session["bookingid"]
    print(id)
    regid= Booking.objects.get(id=id)  
    question = Question.objects.all()
    print(question)
    if request.POST:
        raw_score=0
        rr=Responsemaster.objects.create(bookingid=regid)
        rr.save()
        print(rr)
        res=Responsemaster.objects.get(id=rr.id)
        r=res.id
        print(r)
        for i in question:
                print(i.id)
                nm="point"+str(i.id)
                name=request.POST[nm]
                print(name)
                res=Responsechild.objects.create(respid=rr,question=i.question,response=name)
                res.save()
                if i.id==1 or i.id==7 or i.id==10 or i.id==13 or i.id==17 or i.id==21 or i.id==25 or i.id==29:
                        if name=="1":
                            name="4"
                        elif name=="2":
                            name="3"
                        elif name=="3":
                            name="2"
                        elif name=="4":
                            name="1"
                raw_score+=int(name) 
                print(raw_score) 
                psq=(raw_score-30)/90 
                print(psq)
        level=""
        if psq>=0 and psq<0.34:
            level="Low"
        elif psq>=0.34 and psq<=0.46:
            level="Moderate"
        elif psq>0.46:
            level="High"
        print(level)

        resp=Responsemaster.objects.get(id=r)
        resp.level=level
        resp.save()
        request.session["level"]=level
        return redirect("/patientquestionnaire")
    return render(request, "patientquestion.html", {"question": question})
######################################################################
#                           LOAD PATIENT QUESTIONNAIRE
######################################################################
def patientquestionnaire(request):
    """ 
        The function to load questionnaire
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    if request.POST:
        sleep=request.POST.get("txtSleep")
        energy=request.POST.get("txtEnergy")
        prefer=request.POST.get("txtPrefer")
        irritate=request.POST.get("txtIrritate")
        yourself=request.POST.get("txtYourself")
        relaxed=request.POST.get("txtRelaxed")

        id = request.session["bookingid"]
        
        regid= Booking.objects.get(id=id)  
        s=Survey.objects.create(bookingid=regid)
        s.save()
        q=Questionnaire.objects.create(surveyid=s,question="How is your sleep?",answer=sleep)
        q.save()
        q=Questionnaire.objects.create(surveyid=s,question="How is your energy?",answer=energy)
        q.save()
        q=Questionnaire.objects.create(surveyid=s,question="What do you prefer - to stay at home rather than going out and doing new things? Why?",answer=prefer)
        q.save()
        q=Questionnaire.objects.create(surveyid=s,question="What makes you irritated? Why?",answer=irritate)
        q.save()
        q=Questionnaire.objects.create(surveyid=s,question="How do you feel about yourself?",answer=yourself)
        q.save()
        q=Questionnaire.objects.create(surveyid=s,question="What makes you relaxed?",answer=relaxed)
        q.save()

        pos=0
        neu=0
        neg=0

        sleepsenti=datreeINPUT(sleep)
        print(sleepsenti)

        if sleepsenti=="Positive":
            pos+=1
        elif sleepsenti=="Negative":
            neg+=1
        elif sleepsenti=="Neutral":
            neu+=1

        if datreeINPUT(energy)=="Positive":
            pos+=1
        elif datreeINPUT(energy)=="Negative":
            neg+=1
        elif datreeINPUT(energy)=="Neutral":
            neu+=1

        if datreeINPUT(prefer)=="Positive":
            pos+=1
        elif datreeINPUT(prefer)=="Negative":
            neg+=1
        elif datreeINPUT(prefer)=="Neutral":
            neu+=1

        if datreeINPUT(irritate)=="Positive":
            pos+=1
        elif datreeINPUT(irritate)=="Negative":
            neg+=1
        elif datreeINPUT(irritate)=="Neutral":
            neu+=1

        if datreeINPUT(yourself)=="Positive":
            pos+=1
        elif datreeINPUT(yourself)=="Negative":
            neg+=1
        elif datreeINPUT(yourself)=="Neutral":
            neu+=1

        if datreeINPUT(relaxed)=="Positive":
            pos+=1
        elif datreeINPUT(relaxed)=="Negative":
            neg+=1
        elif datreeINPUT(relaxed)=="Neutral":
            neu+=1

        print(pos,neg,neu)
        
        big=""
        if int(pos)>int(neu) and int(pos)>int(neg) :
            big="Positive"
        elif int(neu)>int(pos) and int(neu)>int(neg):
            big="Neutral"
        else:
            big="Negative"

        print(big)
        request.session["sentiment"]=big
        request.session["positive"]=pos
        request.session["negative"]=neg
        request.session["neutral"]=neu

        print(s)
        
        sur=Survey.objects.get(id=s.id)
        sur.positive=pos
        sur.negative=neg
        sur.neutral=neu
        sur.sentiment=big
        sur.save()
        
        print("updated")
        if big=="positive":
            Booking.objects.get(id=id).delete()
        messages.info(request, 'Test completed successfully')
        return redirect("/patientresult")

    return render(request, "patientquestionnaire.html")
######################################################################
#                           LOAD PATIENT QUESTION PAGE
######################################################################
def patientquestion1(request):
    """ 
        The function to load question page of the patient. 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
     
    question = Question.objects.all()
    print(question)
    if request.POST:
        raw_score=0
        
        for i in question:
                print(i.id)
                nm="point"+str(i.id)
                name=request.POST[nm]
                print(name)
                
                if i.id==1 or i.id==7 or i.id==10 or i.id==13 or i.id==17 or i.id==21 or i.id==25 or i.id==29:
                        if name=="1":
                            name="4"
                        elif name=="2":
                            name="3"
                        elif name=="3":
                            name="2"
                        elif name=="4":
                            name="1"
                raw_score+=int(name) 
                print(raw_score) 
                psq=(raw_score-30)/90 
                print(psq)
        level=""
        if psq>=0 and psq<0.34:
            level="Low"
        elif psq>=0.34 and psq<=0.46:
            level="Moderate"
        elif psq>0.46:
            level="High"
        print(level)
        request.session["level"]=level
        return redirect("/patientquestionnaire1")
    return render(request, "patientquestion1.html", {"question": question})
######################################################################
#                           LOAD PATIENT QUESTIONNAIRE
######################################################################
def patientquestionnaire1(request):
    """ 
        The function to load questionnaire
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    if request.POST:
        sleep=request.POST.get("txtSleep")
        energy=request.POST.get("txtEnergy")
        prefer=request.POST.get("txtPrefer")
        irritate=request.POST.get("txtIrritate")
        yourself=request.POST.get("txtYourself")
        relaxed=request.POST.get("txtRelaxed")

        
        pos=0
        neu=0
        neg=0

        sleepsenti=datreeINPUT(sleep)
        print(sleepsenti)

        if sleepsenti=="Positive":
            pos+=1
        elif sleepsenti=="Negative":
            neg+=1
        elif sleepsenti=="Neutral":
            neu+=1

        if datreeINPUT(energy)=="Positive":
            pos+=1
        elif datreeINPUT(energy)=="Negative":
            neg+=1
        elif datreeINPUT(energy)=="Neutral":
            neu+=1

        if datreeINPUT(prefer)=="Positive":
            pos+=1
        elif datreeINPUT(prefer)=="Negative":
            neg+=1
        elif datreeINPUT(prefer)=="Neutral":
            neu+=1

        if datreeINPUT(irritate)=="Positive":
            pos+=1
        elif datreeINPUT(irritate)=="Negative":
            neg+=1
        elif datreeINPUT(irritate)=="Neutral":
            neu+=1

        if datreeINPUT(yourself)=="Positive":
            pos+=1
        elif datreeINPUT(yourself)=="Negative":
            neg+=1
        elif datreeINPUT(yourself)=="Neutral":
            neu+=1

        if datreeINPUT(relaxed)=="Positive":
            pos+=1
        elif datreeINPUT(relaxed)=="Negative":
            neg+=1
        elif datreeINPUT(relaxed)=="Neutral":
            neu+=1

        print(pos,neg,neu)
        
        big=""
        if int(pos)>int(neu) and int(pos)>int(neg) :
            big="Positive"
        elif int(neu)>int(pos) and int(neu)>int(neg):
            big="Neutral"
        else:
            big="Negative"

        print(big)
        request.session["sentiment"]=big
        request.session["positive"]=pos
        request.session["negative"]=neg
        request.session["neutral"]=neu

        

        messages.info(request, 'Test completed successfully')
        return redirect("/patientresult")

    return render(request, "patientquestionnaire1.html")
######################################################################
#                           LOAD PATIENT SEARCH PAGE
######################################################################
def patientresult(request):
    """ 
        The function to search page of the patient 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    return render(request,"patientresult.html")
######################################################################
#                           LOAD PATIENT SEARCH PAGE
######################################################################


def patientsearchdoctor(request):
    """ 
        The function to search page of the patient 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    doctor = Therapist.objects.all()
    return render(request, "patientsearchdoctor.html", {"doctor": doctor})
######################################################################
#                           LOAD PATIENT BOOKING
######################################################################


def patientbookingdate(request):
    """ 
        The function to add booking
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    did = request.GET.get("id")
    doc=Therapist.objects.get(id=did)
    id = request.session["id"]
    rid=Registration.objects.get(id=id)
    if request.POST:
        date = request.POST["txtDate"]
        try:
            b = Booking.objects.create(regid=rid, therapistid=doc, bookingdate=date,status='Booked')
            b.save()
            request.session['bookingid']=b.id
        except:
            messages.info(request, 'Sorry some error occured')
        else:
            # messages.info(request, 'Booking successfull')
            return redirect("/patientquestion")
    return render(request, "patientbookingdate.html")
######################################################################
#                           LOAD PATIENT BOOKING
######################################################################
def patientbookings(request):
    """ 
        The function to view booking
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id = request.session["id"]
    rid=Registration.objects.get(id=id)
    booking=Booking.objects.filter(regid=rid)
    return render(request, "patientbookings.html",{"booking":booking})
######################################################################
#                           LOAD PATIENT SEARCH PAGE
######################################################################
def patientbookingdetails(request):
    """ 
        The function to search page of the patient 
        -----------------------------------------------
        Parameters: 
            HTTP request 

        Returns: 
            html page
    """
    id=request.GET.get("id")
    booking=Booking.objects.get(id=id)
    responsemaster=Responsemaster.objects.get(bookingid=id)
    print(responsemaster)
    rid=responsemaster.id
    print(rid)
    responsechild=Responsechild.objects.filter(respid=rid)
    survey=Survey.objects.get(bookingid=id)
    questionnaire=Questionnaire.objects.filter(surveyid=survey.id)
    return render(request, "patientbookingdetails.html",{"booking":booking,"responsemaster":responsemaster, "survey":survey,"questionnaire":questionnaire,"responsechild":responsechild})