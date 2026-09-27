"""Page-aware academic PDF layout with overflow checks and traceable captions."""
import html,io,json,re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph,Spacer,Table,TableStyle,Image,Preformatted,Frame,KeepTogether
from reportlab.pdfgen import canvas
from pypdf import PdfReader,PdfWriter

ROOT=Path(__file__).resolve().parents[1]
W=483;PAGES=[];FIGURES=[];TABLES=[];CODES=[]
for name,file in [('TimesVN','times.ttf'),('TimesVN-Bold','timesbd.ttf'),('TimesVN-Italic','timesi.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('TimesVN',normal='TimesVN',bold='TimesVN-Bold',italic='TimesVN-Italic',boldItalic='TimesVN-Bold')
INK=colors.HexColor('#182e42');BLUE=colors.HexColor('#214f70');PALE=colors.HexColor('#eef3f6')
ST={
 'body':ParagraphStyle('body',fontName='TimesVN',fontSize=12,leading=17.1,alignment=TA_JUSTIFY,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='TimesVN',fontSize=10.5,leading=14.1,spaceAfter=8),
 'title':ParagraphStyle('title',fontName='TimesVN-Bold',fontSize=16,leading=21,textColor=INK,spaceAfter=15),
 'sub':ParagraphStyle('sub',fontName='TimesVN-Bold',fontSize=12,leading=16,spaceBefore=5,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='TimesVN',fontSize=10.2,leading=13.1),
 'caption':ParagraphStyle('caption',fontName='TimesVN-Italic',fontSize=10.1,leading=13.2,spaceBefore=5,spaceAfter=10,alignment=TA_CENTER),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=8.8,leading=11.5,spaceAfter=8),
 'toc':ParagraphStyle('toc',fontName='TimesVN',fontSize=10.3,leading=13.5),
}

def page(title,chapter=None):
    PAGES.append(dict(title=title,chapter=chapter,blocks=[]))
    return len(PAGES)+1
def add(flow):PAGES[-1]['blocks'].append(flow)
def rich(text):
    # Preserve intentional markup/entities while escaping ordinary ampersands.
    return re.sub(r'&(?!#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]+;)', '&amp;', text)
def p(text,small=False):add(Paragraph(rich(text),ST['small' if small else 'body']))
def sub(text):add(Paragraph(rich(text),ST['sub']))
def gap(h=7):add(Spacer(1,h))
def table(headers,rows,widths=None,caption=None,small=False):
    style=ST['cell'] if not small else ST['toc']
    values=[[Paragraph('<b>'+html.escape(str(x))+'</b>',style) for x in headers]]
    values += [[Paragraph(html.escape(str(x)),style) for x in row] for row in rows]
    t=Table(values,colWidths=widths or [W/len(headers)]*len(headers),hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('LINEBELOW',(0,0),(-1,0),.7,BLUE),('LINEBELOW',(0,1),(-1,-1),.22,colors.HexColor('#d5dee4')),
                          ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                          ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    add(t)
    if caption:
        TABLES.append((len(TABLES)+1,caption,len(PAGES)))
        add(Paragraph(rich(f'Bảng {len(TABLES)}. '+caption),ST['caption']))
    else:gap()
def code(text,caption):
    import inspect
    text=inspect.cleandoc(text).strip('\n')
    lines=text.splitlines()
    assert max(map(len,lines),default=0)<=93,('Code line too long',caption,max(map(len,lines)))
    add(Preformatted(text,ST['code']))
    CODES.append((len(CODES)+1,caption,len(PAGES)))
    add(Paragraph(rich(f'Mã {len(CODES)}. '+caption),ST['caption']))
def fig(path,caption,width=W,maxheight=285):
    target=ROOT/'figures'/path
    im=Image(str(target));scale=min(width/im.imageWidth,maxheight/im.imageHeight)
    im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
    FIGURES.append((len(FIGURES)+1,caption,len(PAGES)))
    add(im);add(Paragraph(rich(f'Hình {len(FIGURES)}. '+caption),ST['caption']))
def math(lines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if isinstance(lines,str):lines=[lines]
    with plt.rc_context({'mathtext.fontset':'stix'}):
        f=plt.figure(figsize=(6.7,.46*len(lines)+.2),facecolor='#eef3f6')
        for i,line in enumerate(lines):f.text(.5,1-(i+.6)/(len(lines)+.25),'$'+line+'$',ha='center',va='center',fontsize=14,color='#182e42')
        f.canvas.draw()
        for text in f.texts:
            extent=text.get_window_extent(f.canvas.get_renderer())
            if extent.width>f.bbox.width*.95:text.set_fontsize(14*f.bbox.width*.95/extent.width)
        b=io.BytesIO();f.savefig(b,format='png',dpi=210,facecolor=f.get_facecolor());plt.close(f);b.seek(0)
    add(Image(b,width=W,height=(.46*len(lines)+.2)*72));gap(10)
def link(url,label=None):return f'<a href="{html.escape(url,quote=True)}" color="#214f70">{html.escape(label or url)}</a>'
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def pct(v):return f'{100*float(v):.2f}'.replace('.',',')
def dec(v,n=4):return f'{float(v):.{n}f}'.replace('.',',')
def mean_sd(mean,sd,percent=True):
    scale=100 if percent else 1
    digits=2 if percent else 4
    return f'{float(mean)*scale:.{digits}f} ± {float(sd)*scale:.{digits}f}'.replace('.',',')

def render(target,expected_pages=96):
    assert len(PAGES)+1==expected_pages,('Unexpected page count',len(PAGES)+1,expected_pages)
    tmp=ROOT/'report/_expanded_body.pdf';c=canvas.Canvas(str(tmp),pagesize=A4)
    c.setTitle('Assignment 06 - RNN - Nguyễn Ngọc Hoàng Nam');c.setAuthor('Nguyễn Ngọc Hoàng Nam')
    layout=[]
    for index,item in enumerate(PAGES,1):
        c.setFont('TimesVN',9);c.setFillColor(colors.HexColor('#5b6269'))
        c.drawString(59,A4[1]-32,'PTIT | PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH')
        c.drawRightString(A4[0]-53,A4[1]-32,'ASSIGNMENT 06')
        c.setStrokeColor(colors.HexColor('#bccbd5'));c.line(59,A4[1]-40,A4[0]-53,A4[1]-40)
        c.drawString(59,28,'Nguyễn Ngọc Hoàng Nam - B23DCCN585');c.drawRightString(A4[0]-53,28,str(index))
        c.bookmarkPage('page'+str(index))
        if item['chapter'] is not None:c.addOutlineEntry(item['title'],'page'+str(index),level=0,closed=False)
        story=[Paragraph(rich(item['title']),ST['title'])]+item['blocks']
        # Measure flowables on the same width before drawing to catch content overflow.
        heights=[]
        for flow in story:
            _,h=flow.wrap(W,10000);heights.append(h+flow.getSpaceBefore()+flow.getSpaceAfter())
        frame=Frame(59,52,W,A4[1]-112,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        frame.addFromList(story,c)
        if story:raise RuntimeError(f'Overflow on PDF page {index+1}: {item["title"]}; remaining {len(story)} blocks')
        layout.append(dict(pdf_page=index+1,title=item['title'],estimated_content_height=round(sum(heights),1)))
        c.showPage()
    c.save()
    writer=PdfWriter();writer.append(str(ROOT/'report/cover.pdf'));writer.append(str(tmp))
    writer.add_metadata({'/Title':'Assignment 06 - RNN - Nguyễn Ngọc Hoàng Nam - B23DCCN585',
                         '/Author':'Nguyễn Ngọc Hoàng Nam','/Subject':'Retailrocket; S&P 500; Simple RNN; PyTorch; Keras'})
    with Path(target).open('wb') as f:writer.write(f)
    tmp.unlink()
    assert len(PdfReader(target).pages)==expected_pages
    (ROOT/'results/report_layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Created {expected_pages}-page report:',target)
