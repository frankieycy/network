import os,glob

progressBars = ['-','\\','|','/']

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

class emailHandler():
    # email notifier of running process
    def __init__(self, emailFrom, emailPw, emailTo):
        import smtplib
        self.emailFrom = emailFrom
        self.emailTo = emailTo

        self.server = smtplib.SMTP('smtp.gmail.com:587')
        self.server.ehlo()
        self.server.starttls()
        self.server.login(emailFrom,emailPw)

    def setEmailTitle(self, emailTitle):
        self.emailTitle = emailTitle
        self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
            'From: %s'%self.emailFrom,
            'To: %s'%self.emailTo,
            'Subject: (Program email) %s'%self.emailTitle,
            '',
            'process starts'
        ]))

    def sendEmail(self, emailContent):
        self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
            'From: %s'%self.emailFrom,
            'To: %s'%self.emailTo,
            'Subject: Re: (Program email) %s'%self.emailTitle,
            '',
            emailContent
        ]))

    def quitEmail(self):
        self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
            'From: %s'%self.emailFrom,
            'To: %s'%self.emailTo,
            'Subject: (Program email) program completes'
        ]))
        self.server.quit()
