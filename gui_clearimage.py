import wx
app = wx.App()
frame = wx.Frame(None,-1,"Vives' Face Recognizer!")
frame.Show()
menu = wx.Menu()
menuobject = menu.Append(wx.ID_EXIT,'Quit','Quit application')
app.MainLoop()