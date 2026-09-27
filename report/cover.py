"""Cover recreated with the original Assignment 04 border and PTIT logo."""
from report_engine import ROOT
from reportlab.pdfgen import canvas
def create_cover():
    c=canvas.Canvas(str(ROOT/"report/cover.pdf"),pagesize=(596,842))
    c.drawImage(str(ROOT/"report/assets/border.png"),62.539,141.807,width=501,height=684,mask="auto")
    c.drawImage(str(ROOT/"report/assets/logo.png"),260.811,577.606,width=102,height=132.75,mask="auto")
    def center(text,y,size=14,font="TimesVN-Bold"):
        c.setFont(font,size);c.drawCentredString(313,y,text)
    center("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG",772,14)
    center("BỘ MÔN PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH",748,12)
    c.line(247,727,374,727)
    center("PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH",501,18)
    center("ASSIGNMENT 06",469,18)
    center("MẠNG NƠ-RON HỒI TIẾP (RNN)",438,14)
    items=[("Giảng viên hướng dẫn","TRẦN ĐÌNH QUẾ"),
           ("Họ và tên sinh viên","NGUYỄN NGỌC HOÀNG NAM"),
           ("Mã sinh viên","B23DCCN585"),("Lớp","D23CQCN11-B"),("Nhóm","05")]
    for i,(label,value) in enumerate(items):
        c.setFont("TimesVN-Bold",12)
        c.drawString(169,388-17.5*i,label)
        c.drawString(325,388-17.5*i,": "+value)
    center("Hà Nội - 2026",175,12,"TimesVN-Italic")
    c.save()
