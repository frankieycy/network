import os,glob
import smtplib
from pandas import read_csv

progressBars = ['-','\\','|','/']

def loadcsv(f):
    return read_csv(f).values

def setUpFolder(folder, format=None):
    # create folder if unexisting or clear folder
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if format: files = glob.glob(os.path.join(folder,format))
    else: files = glob.glob(os.path.join(folder,'*'))
    for f in files:
        os.remove(f)

def showProgress(currSteps, totSteps):
    # report program progress
    print(' %c %.2f %%\r'%(progressBars[currSteps%4],100.*currSteps/totSteps),end='')

class emailHandler:
    # email notifier of running process
    def __init__(self, emailFrom, emailPw, emailTo):
        self.emailFrom = emailFrom
        self.emailPw = emailPw
        self.emailTo = emailTo
        self.connectServer()

    def connectServer(self):
        self.server = smtplib.SMTP('smtp.gmail.com:587')
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.emailFrom,self.emailPw)

    def setEmailTitle(self, emailTitle):
        self.emailTitle = emailTitle
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: (Program email) %s'%self.emailTitle,
                '',
                'process starts'
            ]))
        except:
            self.connectServer()
            self.setEmailTitle(emailTitle)

    def sendEmail(self, emailContent):
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: Re: (Program email) %s'%self.emailTitle,
                '',
                emailContent
            ]))
        except:
            self.connectServer()
            self.sendEmail(emailContent)

    def quitEmail(self):
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: (Program email) program completes'
            ]))
            self.server.quit()
        except:
            self.connectServer()
            self.quitEmail()
