from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox


class FindDialog(Toplevel):
    def __init__(self,app,edt,replace=False):
        app.root.bind_all("<Escape>", self.closewin)
        self.app = app
        self.edt = edt
        Toplevel.__init__(self,app.root)
        #self.root = Toplevel(app.root)
        self.transient(app.root)
        self.findstring = StringVar()
        self.findstring.trace("w",self.find_onchange)
        
        fieldsFrame = Frame(self)
        frmfind =  LabelFrame(fieldsFrame,text="Find", padx=5, pady=5)        
        fieldsFrame.pack(side=LEFT,fill=Y)
        row = Frame(fieldsFrame)
        Label(row,text="Find").pack()
        
        self.findentry = Entry(row,textvariable=self.findstring)
        self.findentry.pack()
        row.pack()

        if replace:
            row = Frame(fieldsFrame)
            Label(row,text="Replace").pack()
            self.replacetxt = StringVar()
            Entry(row,textvariable=self.replacetxt).pack()
            row.pack()
        
        buttonswFrame = Frame(self)
        Button(buttonswFrame,text="Find next",command=self.search_next).pack(side=TOP)
        Button(buttonswFrame,text="Find previous",command=self.search_previous).pack(side=TOP)
        if replace:
            Button(buttonswFrame,text="Replace",command=self.replace).pack(side=TOP)
            Button(buttonswFrame,text="Replace all",command=self.replace_all).pack(side=TOP)
        buttonswFrame.pack(side=RIGHT,fill=Y)
        
        self.grab_set()
        self.findentry.focus_set()
        app.root.wait_window(self)
        self.edt.tag_delete("search")
        self.edt.focus_set()
                
    def closewin(self,*arg):
        self.destroy()
            

    def find_onchange(self,*arg):
        self.currmatch = None
        self.edt.tag_remove("sel",1.,END)
        self.edt.tag_delete("search")
        self.edt.tag_configure("search", background="green")
        self.edt.tag_raise("sel") 

        
        targettxt = self.findstring.get()
        print("looking for " + targettxt)
        if targettxt == "":
            return
        countVar = StringVar()
        start = "1.0"
        pos = self.edt.search(targettxt, start, stopindex=END, count=countVar)
        firstpos = pos
        while pos != "":            
            end = "%s + %sc" % (pos, countVar.get())
            print(pos,end)
            self.edt.tag_add("search", pos, end)               
            pos = self.edt.search(targettxt,end, stopindex=END, count=countVar)
            
        if firstpos:
            self.currmatch = 0
            self.edt.mark_set("insert",firstpos)
            self.edt.tag_add("sel",self.edt.tag_ranges("search")[0],self.edt.tag_ranges("search")[1])
        else:
            messagebox.showinfo("Find","There are no matches.")

        
    def search_next(self):
        if self.currmatch is None:
            messagebox.showinfo("Find","There are no matches.")
            return
    
        self.currmatch = self.currmatch + 1
        if 2*self.currmatch  == len(self.edt.tag_ranges("search")):
            messagebox.showinfo("Find next","This is the last match")
            self.currmatch = self.currmatch -1
            return

        print(self.edt.tag_ranges("search")[2*self.currmatch])
        self.edt.mark_set("insert", self.edt.tag_ranges("search")[2*self.currmatch] )
        self.edt.see(self.edt.tag_ranges("search")[2*self.currmatch])
        self.edt.tag_remove("sel",1.,END)
        self.edt.tag_add("sel",self.edt.tag_ranges("search")[2*self.currmatch],self.edt.tag_ranges("search")[2*self.currmatch+1])
        
        

    def search_previous(self):
        if self.currmatch is None:
            messagebox.showinfo("Find","There are no matches.")
            return
        if self.currmatch  == 0:
            messagebox.showinfo("Find next","This is the first match")
            return
        self.currmatch = self.currmatch - 1
        print(self.edt.tag_ranges("search")[2*self.currmatch])
        self.edt.mark_set("insert", self.edt.tag_ranges("search")[2*self.currmatch] )
        self.edt.see(self.edt.tag_ranges("search")[2*self.currmatch])
        self.edt.tag_remove("sel",1.,END)
        self.edt.tag_add("sel",self.edt.tag_ranges("search")[2*self.currmatch],self.edt.tag_ranges("search")[2*self.currmatch+1])

        
    def replace(self):
        if self.currmatch is None:
            messagebox.showinfo("Find","There are no matches.")
            return
        begin = self.edt.tag_ranges("search")[2*self.currmatch]
        end =   self.edt.tag_ranges("search")[2*self.currmatch+1]
        self.edt.delete(begin,end)
        self.edt.insert(begin,self.replacetxt.get())
        if self.currmatch >0:
            self.currmatch = self.currmatch  - 1
        self.search_next()


    def replace_all(self):
        if self.currmatch is None:
            messagebox.showinfo("Find","There are no matches.")
        coordinates = []
        lpos = list(self.edt.tag_ranges("search"))
        lpos.reverse()
        
        while lpos:
            coordinates.append([lpos.pop(),lpos.pop()])
            
        for begin,end in coordinates:
            self.edt.delete(begin,end)
            self.edt.insert(begin,self.replacetxt.get())
                            
        messagebox.showinfo("Replace all","There are no more occurrences")            
        pass
